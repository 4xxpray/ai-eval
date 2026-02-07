package agent

import (
	"context"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

func TestToolSelectionEvaluator(t *testing.T) {
	t.Parallel()

	e := &ToolSelectionEvaluator{}

	{
		res, err := e.Evaluate(context.Background(), "", map[string]any{
			"expected_tools": []string{"search", "calculator"},
			"tool_calls": []llm.ToolUse{
				{Name: "search", Input: map[string]any{"q": "x"}},
				{Name: "calculator", Input: map[string]any{"expression": "1+1"}},
			},
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil {
			t.Fatalf("Evaluate: nil result")
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("got passed=%v score=%v", res.Passed, res.Score)
		}
	}

	{
		res, err := e.Evaluate(context.Background(), "", map[string]any{
			"expected_tools": []string{"search", "calculator"},
			"tool_calls": []llm.ToolUse{
				{Name: "search", Input: map[string]any{"q": "x"}},
			},
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil {
			t.Fatalf("Evaluate: nil result")
		}
		if res.Passed || res.Score != 0.5 {
			t.Fatalf("got passed=%v score=%v want false/0.5", res.Passed, res.Score)
		}
	}
}
