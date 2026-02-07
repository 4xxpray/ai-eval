package evaluator

import (
	"context"
	"fmt"
	"regexp"
)

// RegexEvaluator checks that the response matches regex patterns.
type RegexEvaluator struct{}

// Name returns the evaluator identifier.
func (RegexEvaluator) Name() string {
	return "regex"
}

// Evaluate matches the response against regex pattern(s).
// Accepts either a single string or a slice of strings.
func (RegexEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	patterns, err := toStringSlice(expected)
	if err != nil {
		return nil, fmt.Errorf("regex: %w", err)
	}

	if len(patterns) == 0 {
		return &Result{
			Passed:  true,
			Score:   1.0,
			Message: "no patterns to match",
		}, nil
	}

	matched := 0
	var missing []string
	for _, pattern := range patterns {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("regex: compile %q: %w", pattern, err)
		}
		if re.MatchString(response) {
			matched++
		} else {
			missing = append(missing, pattern)
		}
	}

	total := len(patterns)
	score := float64(matched) / float64(total)
	passed := matched == total

	details := map[string]any{
		"matched": matched,
		"total":   total,
	}
	if !passed {
		details["missing"] = missing
	}

	return &Result{
		Passed:  passed,
		Score:   score,
		Message: fmt.Sprintf("matched %d/%d patterns", matched, total),
		Details: details,
	}, nil
}

func toStringSlice(v any) ([]string, error) {
	switch val := v.(type) {
	case string:
		return []string{val}, nil
	case []string:
		return val, nil
	case []any:
		out := make([]string, 0, len(val))
		for i, elem := range val {
			s, ok := elem.(string)
			if !ok {
				return nil, fmt.Errorf("expected[%d]: string, got %T", i, elem)
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected string or []string, got %T", v)
	}
}
