package benchmark

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

const defaultGSM8KPath = "data/benchmark/gsm8k.jsonl"

type GSM8KDataset struct {
	SampleSize int
}

type gsm8kRow struct {
	ID       string `json:"id,omitempty"`
	TaskID   string `json:"task_id,omitempty"`
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

func (d *GSM8KDataset) Name() string { return "gsm8k" }

func (d *GSM8KDataset) Description() string {
	return "GSM8K grade-school math word problems"
}

func (d *GSM8KDataset) Load(ctx context.Context) ([]Question, error) {
	if ctx == nil {
		return nil, errors.New("gsm8k: nil context")
	}

	path := strings.TrimSpace(os.Getenv("AI_EVAL_GSM8K_PATH"))
	if path == "" {
		path = defaultGSM8KPath
	}

	rows, err := readJSONL[gsm8kRow](ctx, path)
	if err != nil {
		if os.IsNotExist(err) {
			return takeFirstN(defaultGSM8KSample(), d.SampleSize), nil
		}
		return nil, fmt.Errorf("gsm8k: load %q: %w", path, err)
	}

	out := make([]Question, 0, len(rows))
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		qText := strings.TrimSpace(row.Question)
		if qText == "" {
			continue
		}

		id := strings.TrimSpace(row.ID)
		if id == "" {
			id = strings.TrimSpace(row.TaskID)
		}
		if id == "" {
			id = fmt.Sprintf("gsm8k-%d", i+1)
		}

		expected := extractExpectedNumber(row.Answer)
		out = append(out, Question{
			ID:       id,
			Question: qText,
			Answer:   expected,
			Category: "math",
		})
	}

	out = takeFirstN(out, d.SampleSize)
	if len(out) == 0 {
		return takeFirstN(defaultGSM8KSample(), d.SampleSize), nil
	}
	return out, nil
}

func (d *GSM8KDataset) Evaluate(response string, expected any) (float64, error) {
	exp := strings.TrimSpace(fmt.Sprint(expected))
	if exp == "" {
		return 0, errors.New("gsm8k: empty expected answer")
	}
	expNum, ok := parseFloat(exp)
	if !ok {
		return 0, fmt.Errorf("gsm8k: invalid expected number %q", exp)
	}

	gotStr, ok := extractLastNumber(response)
	if !ok {
		return 0, errors.New("gsm8k: could not extract number from response")
	}
	gotNum, ok := parseFloat(gotStr)
	if !ok {
		return 0, fmt.Errorf("gsm8k: invalid predicted number %q", gotStr)
	}

	if almostEqual(expNum, gotNum) {
		return 1, nil
	}
	return 0, nil
}

func extractExpectedNumber(answer string) string {
	s := strings.TrimSpace(answer)
	if idx := strings.LastIndex(s, "####"); idx >= 0 {
		s = strings.TrimSpace(s[idx+4:])
	}
	if n, ok := extractLastNumber(s); ok {
		return n
	}
	return strings.TrimSpace(s)
}

func extractLastNumber(s string) (string, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}

	start := -1
	end := -1
	for i := len(s) - 1; i >= 0; i-- {
		c := s[i]
		if (c >= '0' && c <= '9') || c == '.' || c == ',' {
			end = i + 1
			start = i
			for start > 0 {
				pc := s[start-1]
				if (pc >= '0' && pc <= '9') || pc == '.' || pc == ',' || pc == '-' {
					start--
					continue
				}
				break
			}
			break
		}
	}
	if start < 0 || end < 0 || start >= end {
		return "", false
	}
	raw := strings.TrimSpace(s[start:end])
	raw = strings.ReplaceAll(raw, ",", "")
	raw = strings.Trim(raw, ".")
	if raw == "" || raw == "-" {
		return "", false
	}
	return raw, true
}

func parseFloat(s string) (float64, bool) {
	s = strings.TrimSpace(strings.ReplaceAll(s, ",", ""))
	if s == "" {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	return f, err == nil
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func defaultGSM8KSample() []Question {
	return []Question{
		{
			ID:       "gsm8k-sample-1",
			Category: "math",
			Question: "If you have 3 apples and buy 2 more, how many apples do you have?",
			Answer:   "5",
		},
		{
			ID:       "gsm8k-sample-2",
			Category: "math",
			Question: "A box has 12 candies. You eat 5. How many are left?",
			Answer:   "7",
		},
		{
			ID:       "gsm8k-sample-3",
			Category: "math",
			Question: "John has $10 and buys 3 items that each cost $2. How much money does he have left?",
			Answer:   "4",
		},
	}
}

