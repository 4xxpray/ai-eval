package evaluator

import (
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestToolCallEvaluator_NameMatch(t *testing.T) {
	t.Parallel()

	e := ToolCallEvaluator{
		Expected: []testcase.ToolCallExpect{
			{Name: "search", Required: true},
		},
	}

	res := e.Evaluate([]llm.ToolUse{
		{Name: "search", Input: map[string]any{}},
	})

	if !res.Passed {
		t.Fatalf("Passed: got false want true (Details=%#v)", res.Details)
	}
	if res.Score != 1.0 {
		t.Fatalf("Score: got %v want %v", res.Score, 1.0)
	}
}

func TestToolCallEvaluator_PartialArgsMatch(t *testing.T) {
	t.Parallel()

	e := ToolCallEvaluator{
		Expected: []testcase.ToolCallExpect{
			{
				Name:      "search",
				Required:  true,
				ArgsMatch: map[string]any{"q": "hello", "limit": 10},
			},
			{
				Name:     "search",
				Required: true,
				ArgsMatch: map[string]any{
					"filter": map[string]any{
						"type": "repo",
					},
				},
			},
		},
	}

	res := e.Evaluate([]llm.ToolUse{
		{
			Name:  "search",
			Input: map[string]any{"q": "hello", "limit": float64(10), "extra": true},
		},
		{
			Name: "search",
			Input: map[string]any{
				"filter": map[string]any{
					"type":     "repo",
					"language": "go",
				},
				"q": "x",
			},
		},
	})

	if !res.Passed {
		t.Fatalf("Passed: got false want true (Details=%#v)", res.Details)
	}
	if res.Score != 1.0 {
		t.Fatalf("Score: got %v want %v", res.Score, 1.0)
	}
}

func TestToolCallEvaluator_OrderVerification(t *testing.T) {
	t.Parallel()

	got := []llm.ToolUse{
		{Name: "a", Input: map[string]any{}},
		{Name: "b", Input: map[string]any{}},
	}

	{
		e := ToolCallEvaluator{
			Expected: []testcase.ToolCallExpect{
				{Name: "a", Order: 1, Required: true},
				{Name: "b", Order: 2, Required: true},
			},
		}
		res := e.Evaluate(got)
		if !res.Passed {
			t.Fatalf("order ok: Passed=false (Details=%#v)", res.Details)
		}
		if res.Score != 1.0 {
			t.Fatalf("order ok: Score=%v want 1.0", res.Score)
		}
	}

	{
		e := ToolCallEvaluator{
			Expected: []testcase.ToolCallExpect{
				{Name: "b", Order: 1, Required: true},
			},
		}
		res := e.Evaluate(got)
		if res.Passed {
			t.Fatalf("order mismatch: Passed=true want false")
		}
		if res.Score != 0.0 {
			t.Fatalf("order mismatch: Score=%v want 0.0", res.Score)
		}
	}
}

func TestToolCallEvaluator_RequiredVsOptional(t *testing.T) {
	t.Parallel()

	e := ToolCallEvaluator{
		Expected: []testcase.ToolCallExpect{
			{Name: "a", Required: true},
			{Name: "b", Required: false},
		},
	}

	{
		res := e.Evaluate([]llm.ToolUse{{Name: "a", Input: map[string]any{}}})
		if !res.Passed {
			t.Fatalf("optional missing: Passed=false want true (Details=%#v)", res.Details)
		}
		if res.Score != 0.5 {
			t.Fatalf("optional missing: Score=%v want 0.5", res.Score)
		}
	}

	{
		res := e.Evaluate(nil)
		if res.Passed {
			t.Fatalf("required missing: Passed=true want false")
		}
		if res.Score != 0.0 {
			t.Fatalf("required missing: Score=%v want 0.0", res.Score)
		}
	}
}

func TestToolCallEvaluator_RegexArgsMatch(t *testing.T) {
	t.Parallel()

	e := ToolCallEvaluator{
		Expected: []testcase.ToolCallExpect{
			{
				Name:      "search",
				Required:  true,
				ArgsMatch: map[string]any{"q": "regex:^hello"},
			},
		},
	}

	{
		res := e.Evaluate([]llm.ToolUse{{Name: "search", Input: map[string]any{"q": "hello world"}}})
		if !res.Passed {
			t.Fatalf("regex match: Passed=false want true (Details=%#v)", res.Details)
		}
		if res.Score != 1.0 {
			t.Fatalf("regex match: Score=%v want 1.0", res.Score)
		}
	}

	{
		e := ToolCallEvaluator{
			Expected: []testcase.ToolCallExpect{
				{
					Name:      "search",
					Required:  true,
					ArgsMatch: map[string]any{"q": "regex:^world"},
				},
			},
		}
		res := e.Evaluate([]llm.ToolUse{{Name: "search", Input: map[string]any{"q": "hello world"}}})
		if res.Passed {
			t.Fatalf("regex mismatch: Passed=true want false")
		}
		if res.Score != 0.0 {
			t.Fatalf("regex mismatch: Score=%v want 0.0", res.Score)
		}
	}
}

func TestToolCallEvaluator_MultipleToolCalls(t *testing.T) {
	t.Parallel()

	got := []llm.ToolUse{
		{Name: "search", Input: map[string]any{"q": "one"}},
		{Name: "search", Input: map[string]any{"q": "two"}},
	}

	{
		e := ToolCallEvaluator{
			Expected: []testcase.ToolCallExpect{
				{Name: "search", Required: true, ArgsMatch: map[string]any{"q": "two"}},
				{Name: "search", Required: true, ArgsMatch: map[string]any{"q": "one"}},
			},
		}
		res := e.Evaluate(got)
		if !res.Passed {
			t.Fatalf("multi match: Passed=false want true (Details=%#v)", res.Details)
		}
		if res.Score != 1.0 {
			t.Fatalf("multi match: Score=%v want 1.0", res.Score)
		}
	}

	{
		e := ToolCallEvaluator{
			Expected: []testcase.ToolCallExpect{
				{Name: "search", Required: true, ArgsMatch: map[string]any{"q": "one"}},
				{Name: "search", Required: true, ArgsMatch: map[string]any{"q": "one"}},
			},
		}
		res := e.Evaluate([]llm.ToolUse{{Name: "search", Input: map[string]any{"q": "one"}}})
		if res.Passed {
			t.Fatalf("no reuse: Passed=true want false")
		}
		if res.Score != 0.5 {
			t.Fatalf("no reuse: Score=%v want 0.5", res.Score)
		}
	}
}
