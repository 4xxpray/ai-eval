package testcase

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoadFromFile loads and validates a test suite from a YAML file.
func LoadFromFile(path string) (*TestSuite, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("testcase: read %q: %w", path, err)
	}

	var s TestSuite
	if err := yaml.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("testcase: parse %q: %w", path, err)
	}
	if err := Validate(&s); err != nil {
		return nil, fmt.Errorf("testcase: validate %q: %w", path, err)
	}

	return &s, nil
}

// LoadFromDir loads and validates all test suites from a directory.
func LoadFromDir(dir string) ([]*TestSuite, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("testcase: read dir %q: %w", dir, err)
	}

	var paths []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".yaml" && ext != ".yml" {
			continue
		}
		paths = append(paths, filepath.Join(dir, entry.Name()))
	}
	sort.Strings(paths)

	out := make([]*TestSuite, 0, len(paths))
	for _, path := range paths {
		s, err := LoadFromFile(path)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// Validate checks a test suite for consistency.
func Validate(suite *TestSuite) error {
	if suite == nil {
		return fmt.Errorf("nil suite")
	}
	if strings.TrimSpace(suite.Suite) == "" {
		return fmt.Errorf("suite: missing suite name")
	}
	if strings.TrimSpace(suite.Prompt) == "" {
		return fmt.Errorf("suite: missing prompt reference")
	}
	if len(suite.Cases) == 0 {
		return fmt.Errorf("suite: no cases")
	}

	seenIDs := make(map[string]struct{}, len(suite.Cases))
	for i, c := range suite.Cases {
		id := strings.TrimSpace(c.ID)
		if id == "" {
			return fmt.Errorf("cases[%d]: missing id", i)
		}
		if _, ok := seenIDs[id]; ok {
			return fmt.Errorf("cases[%d] (%s): duplicate id", i, id)
		}
		seenIDs[id] = struct{}{}

		if c.Input == nil {
			return fmt.Errorf("cases[%d] (%s): missing input", i, id)
		}
		if c.Trials < 0 {
			return fmt.Errorf("cases[%d] (%s): trials must be >= 0", i, id)
		}
		if c.MaxSteps < 0 {
			return fmt.Errorf("cases[%d] (%s): max_steps must be >= 0", i, id)
		}
		for j, m := range c.ToolMocks {
			name := strings.TrimSpace(m.Name)
			if name == "" {
				return fmt.Errorf("cases[%d] (%s): tool_mocks[%d]: missing name", i, id, j)
			}
		}

		if err := validateExpected(i, id, c.Expected); err != nil {
			return err
		}
		if len(c.Evaluators) == 0 && expectedEmpty(c.Expected) {
			return fmt.Errorf("cases[%d] (%s): no expected assertions or evaluators", i, id)
		}
		if err := validateEvaluators(i, id, c.Evaluators); err != nil {
			return err
		}
	}
	return nil
}

func validateExpected(caseIndex int, caseID string, expected Expected) error {
	for i, s := range expected.Contains {
		if strings.TrimSpace(s) == "" {
			return fmt.Errorf("cases[%d] (%s): expected.contains[%d]: empty string", caseIndex, caseID, i)
		}
	}
	for i, s := range expected.NotContains {
		if strings.TrimSpace(s) == "" {
			return fmt.Errorf("cases[%d] (%s): expected.not_contains[%d]: empty string", caseIndex, caseID, i)
		}
	}
	for i, pattern := range expected.Regex {
		if _, err := regexp.Compile(pattern); err != nil {
			return fmt.Errorf("cases[%d] (%s): expected.regex[%d]: %v", caseIndex, caseID, i, err)
		}
	}

	seenOrder := make(map[int]struct{}, len(expected.ToolCalls))
	for i, tc := range expected.ToolCalls {
		name := strings.TrimSpace(tc.Name)
		if name == "" {
			return fmt.Errorf("cases[%d] (%s): expected.tool_calls[%d]: missing name", caseIndex, caseID, i)
		}
		if tc.Order < 0 {
			return fmt.Errorf("cases[%d] (%s): expected.tool_calls[%d] (%s): order must be >= 0", caseIndex, caseID, i, name)
		}
		if tc.Order > 0 {
			if _, ok := seenOrder[tc.Order]; ok {
				return fmt.Errorf("cases[%d] (%s): expected.tool_calls: duplicate order %d", caseIndex, caseID, tc.Order)
			}
			seenOrder[tc.Order] = struct{}{}
		}
	}

	return nil
}

func expectedEmpty(expected Expected) bool {
	return expected.ExactMatch == "" &&
		len(expected.Contains) == 0 &&
		len(expected.NotContains) == 0 &&
		len(expected.Regex) == 0 &&
		len(expected.JSONSchema) == 0 &&
		len(expected.ToolCalls) == 0
}

func validateEvaluators(caseIndex int, caseID string, evaluators []EvaluatorConfig) error {
	for i, e := range evaluators {
		typ := strings.TrimSpace(e.Type)
		if typ == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d]: missing type", caseIndex, caseID, i)
		}
		if !isKnownEvaluatorType(typ) {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d]: unknown type %q", caseIndex, caseID, i, typ)
		}
		if e.ScoreThreshold < 0 {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (%s): score_threshold must be >= 0", caseIndex, caseID, i, typ)
		}
		if typ == "llm_judge" && strings.TrimSpace(e.Criteria) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (llm_judge): missing criteria", caseIndex, caseID, i)
		}
		if typ == "llm_judge" && e.ScoreScale < 0 {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (llm_judge): score_scale must be >= 0", caseIndex, caseID, i)
		}
		if typ == "similarity" && strings.TrimSpace(e.Reference) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (similarity): missing reference", caseIndex, caseID, i)
		}
		if typ == "factuality" && strings.TrimSpace(e.GroundTruth) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (factuality): missing ground_truth", caseIndex, caseID, i)
		}
		if typ == "faithfulness" && strings.TrimSpace(e.Context) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (faithfulness): missing context", caseIndex, caseID, i)
		}
		if typ == "relevancy" && strings.TrimSpace(e.Question) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (relevancy): missing question", caseIndex, caseID, i)
		}
		if typ == "precision" && strings.TrimSpace(e.Context) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (precision): missing context", caseIndex, caseID, i)
		}
		if typ == "precision" && strings.TrimSpace(e.Question) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (precision): missing question", caseIndex, caseID, i)
		}
		if typ == "task_completion" && strings.TrimSpace(e.Task) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (task_completion): missing task", caseIndex, caseID, i)
		}
		if typ == "efficiency" && e.MaxSteps < 0 {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (efficiency): max_steps must be >= 0", caseIndex, caseID, i)
		}
		if typ == "efficiency" && e.MaxTokens < 0 {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (efficiency): max_tokens must be >= 0", caseIndex, caseID, i)
		}
		if typ == "hallucination" && strings.TrimSpace(e.GroundTruth) == "" {
			return fmt.Errorf("cases[%d] (%s): evaluators[%d] (hallucination): missing ground_truth", caseIndex, caseID, i)
		}
		if typ == "bias" {
			for j, c := range e.Categories {
				if strings.TrimSpace(c) == "" {
					return fmt.Errorf("cases[%d] (%s): evaluators[%d] (bias): categories[%d]: empty string", caseIndex, caseID, i, j)
				}
			}
		}
	}
	return nil
}

func isKnownEvaluatorType(typ string) bool {
	switch typ {
	case "exact", "contains", "regex", "json_schema", "llm_judge", "similarity", "factuality", "tool_call":
		return true
	case "faithfulness", "relevancy", "precision":
		return true
	case "task_completion", "tool_selection", "efficiency":
		return true
	case "hallucination", "toxicity", "bias":
		return true
	default:
		return false
	}
}
