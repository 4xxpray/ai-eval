package benchmark

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const defaultMMLUPath = "data/benchmark/mmlu.jsonl"

type MMLUDataset struct {
	Subjects   []string
	SampleSize int
}

type mmluRow struct {
	ID       string   `json:"id,omitempty"`
	TaskID   string   `json:"task_id,omitempty"`
	Question string   `json:"question"`
	Choices  []string `json:"choices"`
	Answer   any      `json:"answer"`
	Subject  string   `json:"subject,omitempty"`
	Category string   `json:"category,omitempty"`
}

type mcqExpected struct {
	Answer  any      `json:"answer"`
	Choices []string `json:"choices,omitempty"`
}

func (d *MMLUDataset) Name() string { return "mmlu" }

func (d *MMLUDataset) Description() string {
	return "MMLU (Massive Multitask Language Understanding) multiple-choice benchmark"
}

func (d *MMLUDataset) Load(ctx context.Context) ([]Question, error) {
	if ctx == nil {
		return nil, errors.New("mmlu: nil context")
	}

	path := strings.TrimSpace(os.Getenv("AI_EVAL_MMLU_PATH"))
	if path == "" {
		path = defaultMMLUPath
	}

	rows, err := readJSONL[mmluRow](ctx, path)
	if err != nil {
		if os.IsNotExist(err) {
			return takeFirstN(defaultMMLUSample(), d.SampleSize), nil
		}
		return nil, fmt.Errorf("mmlu: load %q: %w", path, err)
	}

	subjectSet := normalizeStringSet(d.Subjects)
	out := make([]Question, 0, len(rows))
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		subject := strings.TrimSpace(row.Subject)
		if len(subjectSet) > 0 && !subjectSet[strings.ToLower(subject)] {
			continue
		}

		qText := strings.TrimSpace(row.Question)
		if qText == "" {
			continue
		}

		choices := compactStrings(row.Choices)
		expected := mcqExpected{Answer: row.Answer, Choices: choices}

		id := strings.TrimSpace(row.ID)
		if id == "" {
			id = strings.TrimSpace(row.TaskID)
		}
		if id == "" {
			id = fmt.Sprintf("mmlu-%d", i+1)
		}

		category := strings.TrimSpace(row.Category)
		if category == "" {
			category = subject
		}

		out = append(out, Question{
			ID:       id,
			Question: qText,
			Choices:  choices,
			Answer:   expected,
			Category: category,
		})
	}

	out = takeFirstN(out, d.SampleSize)
	if len(out) == 0 {
		return takeFirstN(defaultMMLUSample(), d.SampleSize), nil
	}
	return out, nil
}

func (d *MMLUDataset) Evaluate(response string, expected any) (float64, error) {
	expAns, expChoices := unwrapMCQExpected(expected)
	correctIdx, err := expectedChoiceIndex(expAns, expChoices)
	if err != nil {
		return 0, err
	}

	gotIdx, ok := parseMCQResponse(response, expChoices)
	if !ok {
		return 0, errors.New("mmlu: could not parse model answer")
	}
	if gotIdx == correctIdx {
		return 1, nil
	}
	return 0, nil
}

func unwrapMCQExpected(expected any) (any, []string) {
	switch v := expected.(type) {
	case mcqExpected:
		return v.Answer, v.Choices
	case *mcqExpected:
		if v == nil {
			return nil, nil
		}
		return v.Answer, v.Choices
	default:
		return expected, nil
	}
}

func expectedChoiceIndex(answer any, choices []string) (int, error) {
	if len(choices) == 0 {
		choices = []string{"A", "B", "C", "D"}
	}
	max := len(choices)
	if max > 26 {
		max = 26
	}

	switch v := answer.(type) {
	case int:
		return normalizeIndex(v, max)
	case int64:
		return normalizeIndex(int(v), max)
	case float64:
		return normalizeIndex(int(v), max)
	case string:
		return parseExpectedString(v, choices, max)
	default:
		return -1, fmt.Errorf("mmlu: unsupported expected answer type %T", answer)
	}
}

func normalizeIndex(idx int, max int) (int, error) {
	switch {
	case idx >= 0 && idx < max:
		return idx, nil
	case idx >= 1 && idx <= max:
		return idx - 1, nil
	default:
		return -1, fmt.Errorf("mmlu: expected answer out of range (got %d, max %d)", idx, max)
	}
}

func parseExpectedString(s string, choices []string, max int) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return -1, errors.New("mmlu: empty expected answer")
	}

	if len(s) == 1 {
		c := s[0]
		if c >= 'a' && c <= 'z' {
			c = c - 'a' + 'A'
		}
		if c >= 'A' && c <= 'Z' {
			idx := int(c - 'A')
			if idx >= 0 && idx < max {
				return idx, nil
			}
		}
	}

	if n, err := strconv.Atoi(s); err == nil {
		return normalizeIndex(n, max)
	}

	needle := strings.ToLower(s)
	for i, c := range choices {
		if strings.ToLower(strings.TrimSpace(c)) == needle {
			if i < max {
				return i, nil
			}
		}
	}

	return -1, fmt.Errorf("mmlu: could not parse expected answer %q", s)
}

func parseMCQResponse(response string, choices []string) (int, bool) {
	s := strings.TrimSpace(response)
	if s == "" {
		return -1, false
	}

	max := len(choices)
	if max <= 0 {
		max = 4
	}
	if max > 26 {
		max = 26
	}

	if idx, ok := extractLetterToken(s, max); ok {
		return idx, true
	}
	if idx, ok := extractNumberToken(s, max); ok {
		return idx, true
	}
	if idx, ok := matchChoiceText(s, choices, max); ok {
		return idx, true
	}
	return -1, false
}

func extractLetterToken(s string, max int) (int, bool) {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'a' && c <= 'z' {
			c = c - 'a' + 'A'
		}
		if c < 'A' || c > 'Z' {
			continue
		}
		idx := int(c - 'A')
		if idx < 0 || idx >= max {
			continue
		}

		prevOK := i == 0 || !isAlphaNum(s[i-1])
		nextOK := i+1 == len(s) || !isAlphaNum(s[i+1])
		if prevOK && nextOK {
			return idx, true
		}
	}
	return -1, false
}

func extractNumberToken(s string, max int) (int, bool) {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			continue
		}
		j := i
		for j < len(s) && s[j] >= '0' && s[j] <= '9' {
			j++
		}
		n, err := strconv.Atoi(s[i:j])
		if err != nil {
			continue
		}
		if n >= 1 && n <= max {
			return n - 1, true
		}
		if n >= 0 && n < max {
			return n, true
		}
		i = j - 1
	}
	return -1, false
}

func matchChoiceText(s string, choices []string, max int) (int, bool) {
	if len(choices) == 0 {
		return -1, false
	}
	ls := strings.ToLower(s)
	for i, c := range choices {
		if i >= max {
			return -1, false
		}
		c = strings.ToLower(strings.TrimSpace(c))
		if c == "" {
			continue
		}
		if strings.Contains(ls, c) {
			return i, true
		}
	}
	return -1, false
}

func isAlphaNum(b byte) bool {
	return (b >= 'a' && b <= 'z') ||
		(b >= 'A' && b <= 'Z') ||
		(b >= '0' && b <= '9')
}

func normalizeStringSet(in []string) map[string]bool {
	out := make(map[string]bool)
	for _, s := range in {
		v := strings.ToLower(strings.TrimSpace(s))
		if v == "" {
			continue
		}
		out[v] = true
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func compactStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		out = append(out, s)
	}
	return out
}

func defaultMMLUSample() []Question {
	return []Question{
		{
			ID:       "mmlu-sample-1",
			Category: "misc",
			Question: "Which planet is known as the Red Planet?",
			Choices:  []string{"Earth", "Mars", "Jupiter", "Venus"},
			Answer:   mcqExpected{Answer: "B", Choices: []string{"Earth", "Mars", "Jupiter", "Venus"}},
		},
		{
			ID:       "mmlu-sample-2",
			Category: "math",
			Question: "What is 7 * 6?",
			Choices:  []string{"36", "40", "42", "48"},
			Answer:   mcqExpected{Answer: "C", Choices: []string{"36", "40", "42", "48"}},
		},
		{
			ID:       "mmlu-sample-3",
			Category: "science",
			Question: "Water boils at what temperature at sea level (Celsius)?",
			Choices:  []string{"50", "75", "100", "125"},
			Answer:   mcqExpected{Answer: "C", Choices: []string{"50", "75", "100", "125"}},
		},
	}
}
