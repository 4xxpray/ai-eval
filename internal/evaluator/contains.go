package evaluator

import (
	"context"
	"fmt"
	"strings"
)

// ContainsEvaluator checks that all substrings appear in the response.
type ContainsEvaluator struct{}

// Name returns the evaluator identifier.
func (ContainsEvaluator) Name() string {
	return "contains"
}

// Evaluate checks that all expected substrings are present.
func (ContainsEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	substrings, err := asStringSlice(expected)
	if err != nil {
		return nil, fmt.Errorf("contains: %w", err)
	}

	total := len(substrings)
	if total == 0 {
		return &Result{
			Passed:  true,
			Score:   1.0,
			Message: "no expected substrings",
		}, nil
	}

	matched := 0
	var missing []string
	for _, s := range substrings {
		if strings.Contains(response, s) {
			matched++
			continue
		}
		missing = append(missing, s)
	}

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
		Message: fmt.Sprintf("matched %d/%d", matched, total),
		Details: details,
	}, nil
}

// NotContainsEvaluator checks that substrings are absent from the response.
type NotContainsEvaluator struct{}

// Name returns the evaluator identifier.
func (NotContainsEvaluator) Name() string {
	return "not_contains"
}

// Evaluate checks that all expected substrings are absent.
func (NotContainsEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	substrings, err := asStringSlice(expected)
	if err != nil {
		return nil, fmt.Errorf("not_contains: %w", err)
	}

	total := len(substrings)
	if total == 0 {
		return &Result{
			Passed:  true,
			Score:   1.0,
			Message: "no forbidden substrings",
		}, nil
	}

	notFound := 0
	var found []string
	for _, s := range substrings {
		if strings.Contains(response, s) {
			found = append(found, s)
			continue
		}
		notFound++
	}

	score := float64(notFound) / float64(total)
	passed := len(found) == 0
	details := map[string]any{
		"matched": notFound,
		"total":   total,
	}
	if !passed {
		details["found"] = found
	}

	return &Result{
		Passed:  passed,
		Score:   score,
		Message: fmt.Sprintf("matched %d/%d", notFound, total),
		Details: details,
	}, nil
}

func asStringSlice(expected any) ([]string, error) {
	switch v := expected.(type) {
	case nil:
		return nil, fmt.Errorf("expected list of strings, got nil")
	case string:
		return []string{v}, nil
	case []string:
		return v, nil
	case []any:
		out := make([]string, 0, len(v))
		for i, elem := range v {
			s, ok := elem.(string)
			if !ok {
				return nil, fmt.Errorf("expected[%d]: string, got %T", i, elem)
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected list of strings, got %T", expected)
	}
}
