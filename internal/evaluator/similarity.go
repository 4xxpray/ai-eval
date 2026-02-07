package evaluator

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// SimilarityEvaluator scores semantic similarity against a reference.
type SimilarityEvaluator struct {
	Provider  llm.Provider
	Reference string  // Reference answer to compare against
	MinScore  float64 // Minimum similarity score to pass (0.0-1.0)
}

// Name returns the evaluator identifier.
func (SimilarityEvaluator) Name() string {
	return "similarity"
}

type similarityOutput struct {
	Score     float64 `json:"score"`
	Reasoning string  `json:"reasoning"`
}

// Evaluate uses Claude to score similarity.
func (e *SimilarityEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	if e == nil {
		return nil, errors.New("similarity: nil evaluator")
	}
	if e.Provider == nil {
		return nil, errors.New("similarity: nil llm provider")
	}

	reference := strings.TrimSpace(e.Reference)
	minScore := e.MinScore
	if minScore <= 0 {
		minScore = 0.6
	}
	if minScore < 0 {
		minScore = 0
	}
	if minScore > 1 {
		minScore = 1
	}

	switch v := expected.(type) {
	case nil:
	case string:
		if s := strings.TrimSpace(v); s != "" {
			reference = s
		}
	case map[string]any:
		if raw, ok := v["reference"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("similarity: expected.reference must be string, got %T", raw)
			}
			reference = strings.TrimSpace(s)
		}
		if raw, ok := v["min_score"]; ok {
			f, ok := asFloat(raw)
			if !ok {
				return nil, fmt.Errorf("similarity: expected.min_score must be number, got %T", raw)
			}
			minScore = f
		}
	default:
		return nil, fmt.Errorf("similarity: expected must be string or map[string]any, got %T", expected)
	}

	if minScore <= 0 {
		minScore = 0.6
	}
	if minScore < 0 {
		minScore = 0
	}
	if minScore > 1 {
		minScore = 1
	}
	if reference == "" {
		return nil, errors.New("similarity: missing reference")
	}

	var prompt bytes.Buffer
	prompt.WriteString("You are an expert evaluator. Assess whether the AI response is semantically equivalent to the reference answer.\n\n")
	prompt.WriteString("## Reference Answer\n")
	prompt.WriteString(reference)
	prompt.WriteString("\n\n## AI Response\n")
	prompt.WriteString(response)
	prompt.WriteString("\n\n## Instructions\n")
	prompt.WriteString("Rate semantic similarity on a scale from 0.0 to 1.0.\n")
	prompt.WriteString("- 0.0: Completely different or incorrect\n")
	prompt.WriteString("- 1.0: Semantically equivalent (minor wording differences allowed)\n\n")
	prompt.WriteString("Output ONLY valid JSON in this exact format:\n")
	prompt.WriteString("{\"score\": <number 0.0-1.0>, \"reasoning\": \"<brief explanation>\"}")

	resp, err := e.Provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("similarity: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("similarity: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out similarityOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid similarity output",
			Details: map[string]any{"error": err.Error(), "output": raw},
		}, nil
	}

	score := out.Score
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	passed := score >= minScore
	reasoning := strings.TrimSpace(out.Reasoning)
	if reasoning == "" {
		reasoning = "no reasoning provided"
	}

	return &Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: map[string]any{
			"min_score": minScore,
		},
	}, nil
}
