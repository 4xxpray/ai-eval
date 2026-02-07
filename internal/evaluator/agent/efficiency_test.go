package agent

import (
	"context"
	"testing"
)

func TestEfficiencyEvaluator(t *testing.T) {
	t.Parallel()

	e := EfficiencyEvaluator{}

	{
		res, err := e.Evaluate(context.Background(), "", map[string]any{
			"max_steps":     5,
			"max_tokens":    1000,
			"actual_steps":  3,
			"actual_tokens": 500,
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil {
			t.Fatalf("Evaluate: nil result")
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("got passed=%v score=%v want true/1.0", res.Passed, res.Score)
		}
	}

	{
		res, err := e.Evaluate(context.Background(), "", map[string]any{
			"max_steps":     5,
			"max_tokens":    1000,
			"actual_steps":  10,
			"actual_tokens": 1500,
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil {
			t.Fatalf("Evaluate: nil result")
		}
		if res.Passed {
			t.Fatalf("Passed: got true want false")
		}
		if res.Score <= 0 || res.Score >= 1 {
			t.Fatalf("Score: got %v want between 0 and 1", res.Score)
		}
	}
}
