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

// FaithfulnessEvaluator checks whether the response is grounded in retrieval context.
type FaithfulnessEvaluator struct {
	Client llm.Provider
}

func (FaithfulnessEvaluator) Name() string {
	return "faithfulness"
}

type faithfulnessOutput struct {
	Score             float64  `json:"score"`
	Reasoning         string   `json:"reasoning"`
	UnsupportedClaims []string `json:"unsupported_claims"`
}

func (e *FaithfulnessEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	if e == nil {
		return nil, errors.New("faithfulness: nil evaluator")
	}
	if e.Client == nil {
		return nil, errors.New("faithfulness: nil llm provider")
	}

	contextText := ""
	threshold := 0.8

	switch v := expected.(type) {
	case nil:
	case map[string]any:
		if raw, ok := v["context"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("faithfulness: expected.context must be string, got %T", raw)
			}
			contextText = strings.TrimSpace(s)
		}
		if raw, ok := v["threshold"]; ok {
			f, ok := asFloat(raw)
			if !ok {
				return nil, fmt.Errorf("faithfulness: expected.threshold must be number, got %T", raw)
			}
			threshold = f
		}
	default:
		return nil, fmt.Errorf("faithfulness: expected must be map[string]any, got %T", expected)
	}

	if threshold <= 0 {
		threshold = 0.8
	}
	if threshold > 1 {
		threshold = 1
	}
	if contextText == "" {
		return nil, errors.New("faithfulness: missing context")
	}

	var prompt bytes.Buffer
	prompt.WriteString("You are an expert RAG evaluator. Determine whether the AI response is strictly grounded in the provided retrieval context.\n\n")
	prompt.WriteString("## Retrieval Context\n")
	prompt.WriteString(contextText)
	prompt.WriteString("\n\n## AI Response\n")
	prompt.WriteString(response)
	prompt.WriteString("\n\n## Instructions\n")
	prompt.WriteString("Score faithfulness from 0.0 to 1.0.\n")
	prompt.WriteString("- 0.0: Response is mostly unsupported / hallucinatory\n")
	prompt.WriteString("- 1.0: Every factual claim is supported by the context; no new facts are introduced\n\n")
	prompt.WriteString("Output ONLY valid JSON in this exact format:\n")
	prompt.WriteString("{\"score\": <number 0.0-1.0>, \"reasoning\": \"<brief explanation>\", \"unsupported_claims\": [\"<claim>\", ...]}")

	resp, err := e.Client.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("faithfulness: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("faithfulness: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out faithfulnessOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &evaluator.Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid faithfulness output",
			Details: map[string]any{"error": err.Error(), "output": raw},
		}, nil
	}

	score := clamp01(out.Score)
	passed := score >= threshold
	reasoning := strings.TrimSpace(out.Reasoning)
	if reasoning == "" {
		reasoning = "no reasoning provided"
	}

	details := map[string]any{
		"threshold": threshold,
	}
	if len(out.UnsupportedClaims) > 0 {
		details["unsupported_claims"] = out.UnsupportedClaims
	}

	return &evaluator.Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: details,
	}, nil
}
