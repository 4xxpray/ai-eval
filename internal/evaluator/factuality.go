package evaluator

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// FactualityEvaluator checks factual consistency against ground truth.
type FactualityEvaluator struct {
	Provider    llm.Provider
	GroundTruth string // Known facts to verify against
}

// Name returns the evaluator identifier.
func (FactualityEvaluator) Name() string {
	return "factuality"
}

type factualityOutput struct {
	HasError  bool     `json:"has_error"`
	Errors    []string `json:"errors"`
	Reasoning string   `json:"reasoning"`
}

// Evaluate uses Claude to assess factuality.
func (e *FactualityEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	if e == nil {
		return nil, errors.New("factuality: nil evaluator")
	}
	if e.Provider == nil {
		return nil, errors.New("factuality: nil llm provider")
	}

	groundTruth := strings.TrimSpace(e.GroundTruth)

	switch v := expected.(type) {
	case nil:
	case string:
		if s := strings.TrimSpace(v); s != "" {
			groundTruth = s
		}
	case map[string]any:
		if raw, ok := v["ground_truth"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("factuality: expected.ground_truth must be string, got %T", raw)
			}
			groundTruth = strings.TrimSpace(s)
		}
	default:
		return nil, fmt.Errorf("factuality: expected must be string or map[string]any, got %T", expected)
	}

	if groundTruth == "" {
		return nil, errors.New("factuality: missing ground truth")
	}

	var prompt bytes.Buffer
	prompt.WriteString("You are an expert fact checker. Verify the AI response against the provided ground truth.\n\n")
	prompt.WriteString("## Ground Truth\n")
	prompt.WriteString(groundTruth)
	prompt.WriteString("\n\n## AI Response\n")
	prompt.WriteString(response)
	prompt.WriteString("\n\n## Instructions\n")
	prompt.WriteString("Identify any factual errors or contradictions relative to the ground truth.\n")
	prompt.WriteString("If any factual error exists, set has_error to true.\n\n")
	prompt.WriteString("Output ONLY valid JSON in this exact format:\n")
	prompt.WriteString("{\"has_error\": <true|false>, \"errors\": [\"<error 1>\", \"<error 2>\"], \"reasoning\": \"<brief explanation>\"}")

	resp, err := e.Provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("factuality: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("factuality: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out factualityOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid factuality output",
			Details: map[string]any{"error": err.Error(), "output": raw},
		}, nil
	}

	score := 1.0
	if out.HasError {
		score = 0.0
	}
	passed := !out.HasError
	reasoning := strings.TrimSpace(out.Reasoning)
	if reasoning == "" {
		reasoning = "no reasoning provided"
	}

	details := map[string]any{}
	if len(out.Errors) > 0 {
		details["errors"] = out.Errors
	}

	return &Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: details,
	}, nil
}
