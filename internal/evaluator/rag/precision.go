package rag

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// PrecisionEvaluator scores how much of the retrieval context is relevant to the question.
type PrecisionEvaluator struct {
	Client llm.Provider
}

func (PrecisionEvaluator) Name() string {
	return "precision"
}

type precisionOutput struct {
	Score     float64 `json:"score"`
	Reasoning string  `json:"reasoning"`
}

func (e *PrecisionEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	if e == nil {
		return nil, errors.New("precision: nil evaluator")
	}
	if e.Client == nil {
		return nil, errors.New("precision: nil llm provider")
	}

	contextText := ""
	question := ""
	threshold := 0.0

	switch v := expected.(type) {
	case nil:
	case map[string]any:
		if raw, ok := v["context"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("precision: expected.context must be string, got %T", raw)
			}
			contextText = strings.TrimSpace(s)
		}
		if raw, ok := v["question"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("precision: expected.question must be string, got %T", raw)
			}
			question = strings.TrimSpace(s)
		}
		if raw, ok := v["threshold"]; ok {
			f, ok := asFloat(raw)
			if !ok {
				return nil, fmt.Errorf("precision: expected.threshold must be number, got %T", raw)
			}
			threshold = f
		}
	default:
		return nil, fmt.Errorf("precision: expected must be map[string]any, got %T", expected)
	}

	if threshold < 0 {
		threshold = 0
	}
	if threshold > 1 {
		threshold = 1
	}
	if contextText == "" {
		return nil, errors.New("precision: missing context")
	}
	if question == "" {
		return nil, errors.New("precision: missing question")
	}

	var prompt bytes.Buffer
	prompt.WriteString("You are an expert retrieval evaluator. Assess the precision of the retrieved context for answering the question.\n\n")
	prompt.WriteString("## Question\n")
	prompt.WriteString(question)
	prompt.WriteString("\n\n## Retrieved Context\n")
	prompt.WriteString(contextText)
	prompt.WriteString("\n\n## Instructions\n")
	prompt.WriteString("Score precision from 0.0 to 1.0.\n")
	prompt.WriteString("- 0.0: Context is mostly irrelevant noise\n")
	prompt.WriteString("- 1.0: Context is highly focused; most content is directly useful for answering the question\n\n")
	prompt.WriteString("Output ONLY valid JSON in this exact format:\n")
	prompt.WriteString("{\"score\": <number 0.0-1.0>, \"reasoning\": \"<brief explanation>\"}")

	resp, err := e.Client.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("precision: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("precision: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out precisionOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &evaluator.Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid precision output",
			Details: map[string]any{"error": err.Error(), "output": raw},
		}, nil
	}

	score := clamp01(out.Score)
	passed := true
	if threshold > 0 {
		passed = score >= threshold
	}
	reasoning := strings.TrimSpace(out.Reasoning)
	if reasoning == "" {
		reasoning = "no reasoning provided"
	}

	details := map[string]any{}
	if threshold > 0 {
		details["threshold"] = threshold
	}

	return &evaluator.Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: details,
	}, nil
}
