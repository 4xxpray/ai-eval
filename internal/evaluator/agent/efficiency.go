package agent

import (
	"context"
	"fmt"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
)

// EfficiencyEvaluator checks execution efficiency against step/token budgets.
type EfficiencyEvaluator struct{}

func (EfficiencyEvaluator) Name() string {
	return "efficiency"
}

func (EfficiencyEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	_ = ctx
	_ = response

	maxSteps := 0
	maxTokens := 0
	actualSteps := 0
	actualTokens := 0

	switch v := expected.(type) {
	case nil:
	case map[string]any:
		if raw, ok := v["max_steps"]; ok {
			n, ok := asInt(raw)
			if !ok {
				return nil, fmt.Errorf("efficiency: expected.max_steps must be number, got %T", raw)
			}
			maxSteps = n
		}
		if raw, ok := v["max_tokens"]; ok {
			n, ok := asInt(raw)
			if !ok {
				return nil, fmt.Errorf("efficiency: expected.max_tokens must be number, got %T", raw)
			}
			maxTokens = n
		}
		if raw, ok := v["actual_steps"]; ok {
			n, ok := asInt(raw)
			if !ok {
				return nil, fmt.Errorf("efficiency: expected.actual_steps must be number, got %T", raw)
			}
			actualSteps = n
		}
		if raw, ok := v["actual_tokens"]; ok {
			n, ok := asInt(raw)
			if !ok {
				return nil, fmt.Errorf("efficiency: expected.actual_tokens must be number, got %T", raw)
			}
			actualTokens = n
		}
	default:
		return nil, fmt.Errorf("efficiency: expected must be map[string]any, got %T", expected)
	}

	if maxSteps < 0 || maxTokens < 0 || actualSteps < 0 || actualTokens < 0 {
		return nil, fmt.Errorf("efficiency: values must be >= 0")
	}
	if maxSteps == 0 && maxTokens == 0 {
		return nil, fmt.Errorf("efficiency: missing max_steps and max_tokens")
	}

	scoreSum := 0.0
	scoreN := 0.0
	passed := true

	stepScore := 0.0
	if maxSteps > 0 {
		if actualSteps <= 0 {
			stepScore = 0.0
			passed = false
		} else if actualSteps <= maxSteps {
			stepScore = 1.0
		} else {
			stepScore = float64(maxSteps) / float64(actualSteps)
			passed = false
		}
		scoreSum += stepScore
		scoreN++
	}

	tokenScore := 0.0
	if maxTokens > 0 {
		if actualTokens <= 0 {
			tokenScore = 0.0
			passed = false
		} else if actualTokens <= maxTokens {
			tokenScore = 1.0
		} else {
			tokenScore = float64(maxTokens) / float64(actualTokens)
			passed = false
		}
		scoreSum += tokenScore
		scoreN++
	}

	score := 0.0
	if scoreN > 0 {
		score = clamp01(scoreSum / scoreN)
	}

	return &evaluator.Result{
		Passed:  passed,
		Score:   score,
		Message: "efficiency evaluated",
		Details: map[string]any{
			"max_steps":     maxSteps,
			"max_tokens":    maxTokens,
			"actual_steps":  actualSteps,
			"actual_tokens": actualTokens,
			"step_score":    stepScore,
			"token_score":   tokenScore,
		},
	}, nil
}
