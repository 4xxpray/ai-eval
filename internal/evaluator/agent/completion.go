package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// TaskCompletionEvaluator scores whether an agent completed the given task.
type TaskCompletionEvaluator struct {
	Client llm.Provider
}

func (TaskCompletionEvaluator) Name() string {
	return "task_completion"
}

type taskCompletionOutput struct {
	Score     float64  `json:"score"`
	Reasoning string   `json:"reasoning"`
	Missing   []string `json:"missing"`
}

func (e *TaskCompletionEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	if e == nil {
		return nil, errors.New("task_completion: nil evaluator")
	}
	if e.Client == nil {
		return nil, errors.New("task_completion: nil llm provider")
	}

	task := ""
	criteria := []string(nil)
	threshold := 0.6

	switch v := expected.(type) {
	case nil:
	case map[string]any:
		if raw, ok := v["task"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("task_completion: expected.task must be string, got %T", raw)
			}
			task = strings.TrimSpace(s)
		}
		if raw, ok := v["criteria"]; ok {
			ss, err := asStringSlice(raw)
			if err != nil {
				return nil, fmt.Errorf("task_completion: expected.criteria: %w", err)
			}
			criteria = ss
		}
		if raw, ok := v["threshold"]; ok {
			f, ok := asFloat(raw)
			if !ok {
				return nil, fmt.Errorf("task_completion: expected.threshold must be number, got %T", raw)
			}
			threshold = f
		}
	default:
		return nil, fmt.Errorf("task_completion: expected must be map[string]any, got %T", expected)
	}

	if threshold <= 0 {
		threshold = 0.6
	}
	if threshold < 0 {
		threshold = 0
	}
	if threshold > 1 {
		threshold = 1
	}
	if task == "" {
		return nil, errors.New("task_completion: missing task")
	}

	var prompt bytes.Buffer
	prompt.WriteString("You are an expert agent evaluator. Judge whether the response completes the task.\n\n")
	prompt.WriteString("## Task\n")
	prompt.WriteString(task)
	if len(criteria) > 0 {
		prompt.WriteString("\n\n## Completion Criteria\n")
		for _, c := range criteria {
			if s := strings.TrimSpace(c); s != "" {
				prompt.WriteString("- ")
				prompt.WriteString(s)
				prompt.WriteString("\n")
			}
		}
	}
	prompt.WriteString("\n## Agent Response\n")
	prompt.WriteString(response)
	prompt.WriteString("\n\n## Instructions\n")
	prompt.WriteString("Score task completion from 0.0 to 1.0.\n")
	prompt.WriteString("- 0.0: Task not completed\n")
	prompt.WriteString("- 1.0: Task fully completed and meets criteria\n\n")
	prompt.WriteString("Output ONLY valid JSON in this exact format:\n")
	prompt.WriteString("{\"score\": <number 0.0-1.0>, \"reasoning\": \"<brief explanation>\", \"missing\": [\"<missing item>\", ...]}")

	resp, err := e.Client.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("task_completion: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("task_completion: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out taskCompletionOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &evaluator.Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid task_completion output",
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
	if len(out.Missing) > 0 {
		details["missing"] = out.Missing
	}

	return &evaluator.Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: details,
	}, nil
}
