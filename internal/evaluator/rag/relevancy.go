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

// RelevancyEvaluator checks whether the response is relevant to the question.
type RelevancyEvaluator struct {
	Client llm.Provider
}

func (RelevancyEvaluator) Name() string {
	return "relevancy"
}

type relevancyOutput struct {
	Score     float64 `json:"score"`
	Reasoning string  `json:"reasoning"`
}

func (e *RelevancyEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	if e == nil {
		return nil, errors.New("relevancy: nil evaluator")
	}
	if e.Client == nil {
		return nil, errors.New("relevancy: nil llm provider")
	}

	question := ""
	threshold := 0.8

	switch v := expected.(type) {
	case nil:
	case map[string]any:
		if raw, ok := v["question"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("relevancy: expected.question must be string, got %T", raw)
			}
			question = strings.TrimSpace(s)
		}
		if raw, ok := v["threshold"]; ok {
			f, ok := asFloat(raw)
			if !ok {
				return nil, fmt.Errorf("relevancy: expected.threshold must be number, got %T", raw)
			}
			threshold = f
		}
	default:
		return nil, fmt.Errorf("relevancy: expected must be map[string]any, got %T", expected)
	}

	if threshold <= 0 {
		threshold = 0.8
	}
	if threshold < 0 {
		threshold = 0
	}
	if threshold > 1 {
		threshold = 1
	}
	if question == "" {
		return nil, errors.New("relevancy: missing question")
	}

	var prompt bytes.Buffer
	prompt.WriteString("You are an expert evaluator. Assess whether the AI response directly answers the question.\n\n")
	prompt.WriteString("## Question\n")
	prompt.WriteString(question)
	prompt.WriteString("\n\n## AI Response\n")
	prompt.WriteString(response)
	prompt.WriteString("\n\n## Instructions\n")
	prompt.WriteString("Score relevancy from 0.0 to 1.0.\n")
	prompt.WriteString("- 0.0: Completely irrelevant or evasive\n")
	prompt.WriteString("- 1.0: Directly answers the question with no major digressions\n\n")
	prompt.WriteString("Output ONLY valid JSON in this exact format:\n")
	prompt.WriteString("{\"score\": <number 0.0-1.0>, \"reasoning\": \"<brief explanation>\"}")

	resp, err := e.Client.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("relevancy: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("relevancy: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out relevancyOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &evaluator.Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid relevancy output",
			Details: map[string]any{"error": err.Error(), "output": raw},
		}, nil
	}

	score := clamp01(out.Score)
	passed := score >= threshold
	reasoning := strings.TrimSpace(out.Reasoning)
	if reasoning == "" {
		reasoning = "no reasoning provided"
	}

	return &evaluator.Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: map[string]any{
			"threshold": threshold,
		},
	}, nil
}
