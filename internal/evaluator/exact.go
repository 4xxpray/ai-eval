package evaluator

import (
	"context"
	"fmt"
)

// ExactEvaluator checks for an exact string match.
type ExactEvaluator struct{}

// Name returns the evaluator identifier.
func (ExactEvaluator) Name() string {
	return "exact"
}

// Evaluate compares the response to the expected string.
func (ExactEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	exp, ok := expected.(string)
	if !ok {
		return nil, fmt.Errorf("exact: expected string, got %T", expected)
	}

	passed := response == exp
	score := 0.0
	if passed {
		score = 1.0
	}

	msg := "not exact match"
	if passed {
		msg = "exact match"
	}

	return &Result{
		Passed:  passed,
		Score:   score,
		Message: msg,
	}, nil
}
