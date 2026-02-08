package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type stubProvider struct {
	fn func(ctx context.Context, req *llm.Request) (*llm.Response, error)
}

func (p *stubProvider) Name() string { return "stub" }

func (p *stubProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if p.fn == nil {
		return nil, nil
	}
	return p.fn(ctx, req)
}

func (p *stubProvider) CompleteWithTools(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
	_ = ctx
	_ = req
	return nil, errors.New("not implemented")
}

func textResponse(s string) *llm.Response {
	return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: s}}}
}

func TestAgentEvaluators_NameAndClamp(t *testing.T) {
	if (TaskCompletionEvaluator{}).Name() != "task_completion" {
		t.Fatalf("task_completion name")
	}
	if (EfficiencyEvaluator{}).Name() != "efficiency" {
		t.Fatalf("efficiency name")
	}
	if (ToolSelectionEvaluator{}).Name() != "tool_selection" {
		t.Fatalf("tool_selection name")
	}

	if clamp01(-1) != 0 {
		t.Fatalf("clamp01(-1)")
	}
	if clamp01(2) != 1 {
		t.Fatalf("clamp01(2)")
	}
	if clamp01(0.5) != 0.5 {
		t.Fatalf("clamp01(0.5)")
	}
}

func TestAgentUtil_Conversions(t *testing.T) {
	t.Parallel()

	t.Run("asInt", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name string
			in   any
			want int
		}{
			{name: "int", in: int(1), want: 1},
			{name: "int8", in: int8(2), want: 2},
			{name: "int16", in: int16(3), want: 3},
			{name: "int32", in: int32(4), want: 4},
			{name: "int64", in: int64(5), want: 5},
			{name: "uint", in: uint(6), want: 6},
			{name: "uint8", in: uint8(7), want: 7},
			{name: "uint16", in: uint16(8), want: 8},
			{name: "uint32", in: uint32(9), want: 9},
			{name: "uint64", in: uint64(10), want: 10},
			{name: "float32", in: float32(11.9), want: 11},
			{name: "float64", in: float64(12.9), want: 12},
			{name: "json_number", in: json.Number("13"), want: 13},
		}
		for _, tt := range tests {
			got, ok := asInt(tt.in)
			if !ok || got != tt.want {
				t.Fatalf("%s: asInt(%T=%v): got=%d ok=%v want=%d,true", tt.name, tt.in, tt.in, got, ok, tt.want)
			}
		}

		if _, ok := asInt(json.Number("x")); ok {
			t.Fatalf("expected asInt(json invalid) false")
		}
		if _, ok := asInt(json.Number("1.5")); ok {
			t.Fatalf("expected asInt(json decimal) false")
		}
		if _, ok := asInt("x"); ok {
			t.Fatalf("expected asInt(string) false")
		}
	})

	t.Run("asFloat", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name string
			in   any
			want float64
		}{
			{name: "float64", in: float64(1.5), want: 1.5},
			{name: "float32", in: float32(1.25), want: 1.25},
			{name: "int", in: int(-1), want: -1},
			{name: "int8", in: int8(-2), want: -2},
			{name: "int16", in: int16(-3), want: -3},
			{name: "int32", in: int32(-4), want: -4},
			{name: "int64", in: int64(-5), want: -5},
			{name: "uint", in: uint(6), want: 6},
			{name: "uint8", in: uint8(7), want: 7},
			{name: "uint16", in: uint16(8), want: 8},
			{name: "uint32", in: uint32(9), want: 9},
			{name: "uint64", in: uint64(10), want: 10},
			{name: "json_number", in: json.Number("3.5"), want: 3.5},
		}
		for _, tt := range tests {
			got, ok := asFloat(tt.in)
			if !ok || got != tt.want {
				t.Fatalf("%s: asFloat(%T=%v): got=%v ok=%v want=%v,true", tt.name, tt.in, tt.in, got, ok, tt.want)
			}
		}

		if _, ok := asFloat(json.Number("x")); ok {
			t.Fatalf("expected asFloat(json invalid) false")
		}
		if _, ok := asFloat("x"); ok {
			t.Fatalf("expected asFloat(string) false")
		}
	})

	{
		if _, err := asStringSlice(nil); err == nil {
			t.Fatalf("expected error")
		}
		if ss, err := asStringSlice("a"); err != nil || len(ss) != 1 || ss[0] != "a" {
			t.Fatalf("ss=%#v err=%v", ss, err)
		}
		if ss, err := asStringSlice([]any{"a", "b"}); err != nil || len(ss) != 2 {
			t.Fatalf("ss=%#v err=%v", ss, err)
		}
		if _, err := asStringSlice([]any{"a", 1}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := asStringSlice(123); err == nil {
			t.Fatalf("expected error")
		}
	}
}

func TestToolNamesFromAny(t *testing.T) {
	{
		_, err := toolNamesFromAny(nil)
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		names, err := toolNamesFromAny([]llm.ToolUse{{Name: "a"}, {Name: " "}, {Name: "b"}})
		if err != nil {
			t.Fatalf("err=%v", err)
		}
		if len(names) != 2 || names[0] != "a" || names[1] != "b" {
			t.Fatalf("names=%#v", names)
		}
	}
	{
		names, err := toolNamesFromAny([]any{
			llm.ToolUse{Name: "a"},
			map[string]any{"name": "b"},
		})
		if err != nil {
			t.Fatalf("err=%v", err)
		}
		if len(names) != 2 || names[0] != "a" || names[1] != "b" {
			t.Fatalf("names=%#v", names)
		}
	}
	{
		_, err := toolNamesFromAny([]any{map[string]any{}})
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := toolNamesFromAny([]any{map[string]any{"name": 1}})
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := toolNamesFromAny([]any{123})
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := toolNamesFromAny("x")
		if err == nil {
			t.Fatalf("expected error")
		}
	}
}

func TestSetKeysSorted(t *testing.T) {
	if got := setKeysSorted(nil); got != nil {
		t.Fatalf("got=%#v", got)
	}
	if got := setKeysSorted(map[string]struct{}{}); got != nil {
		t.Fatalf("got=%#v", got)
	}
	got := setKeysSorted(map[string]struct{}{"b": {}, "a": {}})
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("got=%#v", got)
	}
}

func TestToolSelectionEvaluator_NoExpectedTools(t *testing.T) {
	t.Parallel()

	e := &ToolSelectionEvaluator{}

	{
		res, err := e.Evaluate(context.Background(), "", map[string]any{
			"expected_tools": []string{},
			"tool_calls":     []llm.ToolUse{{Name: "search"}},
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 {
			t.Fatalf("res=%#v", res)
		}
		if got := res.Details["unexpected_tools"].([]string); len(got) != 1 || got[0] != "search" {
			t.Fatalf("unexpected=%#v", got)
		}
	}

	{
		res, err := e.Evaluate(context.Background(), "", map[string]any{
			"expected_tools": []string{},
			"tool_calls":     []llm.ToolUse{{Name: " "}},
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed || res.Score != 1 {
			t.Fatalf("res=%#v", res)
		}
	}
}

func TestToolSelectionEvaluator_ErrorPaths(t *testing.T) {
	t.Parallel()

	e := &ToolSelectionEvaluator{}

	if _, err := e.Evaluate(context.Background(), "", 123); err == nil {
		t.Fatalf("expected error for wrong expected type")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{
		"expected_tools": []any{"a", 1},
		"tool_calls":     []llm.ToolUse{},
	}); err == nil {
		t.Fatalf("expected error for expected_tools type")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{
		"expected_tools": []string{"search"},
	}); err == nil || !strings.Contains(err.Error(), "tool_calls") {
		t.Fatalf("expected tool_calls error, got %v", err)
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{
		"expected_tools": []string{"search"},
		"tool_calls":     []any{map[string]any{"name": 1}},
	}); err == nil {
		t.Fatalf("expected tool_calls parse error")
	}
}

func TestToolSelectionEvaluator_UnexpectedTools(t *testing.T) {
	t.Parallel()

	e := &ToolSelectionEvaluator{}

	res, err := e.Evaluate(context.Background(), "", map[string]any{
		"expected_tools": []string{"search", " "},
		"tool_calls": []llm.ToolUse{
			{Name: "search"},
			{Name: "calculator"},
			{Name: " "},
		},
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 1 {
		t.Fatalf("res=%#v", res)
	}
	unexpected, _ := res.Details["unexpected_tools"].([]string)
	if len(unexpected) != 1 || unexpected[0] != "calculator" {
		t.Fatalf("unexpected=%#v", unexpected)
	}
}

func TestEfficiencyEvaluator_ErrorPaths(t *testing.T) {
	t.Parallel()

	e := EfficiencyEvaluator{}

	if _, err := e.Evaluate(context.Background(), "", 123); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{"max_steps": "x"}); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{"max_tokens": "x"}); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{"max_steps": 1, "actual_steps": "x"}); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{"max_tokens": 1, "actual_tokens": "x"}); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{"max_steps": 0, "max_tokens": 0}); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "", map[string]any{"max_steps": -1}); err == nil {
		t.Fatalf("expected error")
	}

	res, err := e.Evaluate(context.Background(), "", map[string]any{
		"max_steps":    5,
		"actual_steps": 0,
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 {
		t.Fatalf("res=%#v", res)
	}

	res, err = e.Evaluate(context.Background(), "", map[string]any{
		"max_tokens": 10,
	})
	if err != nil {
		t.Fatalf("Evaluate(token): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 {
		t.Fatalf("res=%#v", res)
	}
}

func TestTaskCompletionEvaluator_Evaluate(t *testing.T) {
	t.Run("NilEvaluator", func(t *testing.T) {
		var e *TaskCompletionEvaluator
		_, err := e.Evaluate(context.Background(), "x", nil)
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("NilClient", func(t *testing.T) {
		e := &TaskCompletionEvaluator{}
		_, err := e.Evaluate(context.Background(), "x", map[string]any{"task": "t"})
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ExpectedTypeErrors", func(t *testing.T) {
		e := &TaskCompletionEvaluator{Client: &stubProvider{}}
		if _, err := e.Evaluate(context.Background(), "x", 123); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"task": 1}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"task": "t", "criteria": []any{"a", 1}}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"task": "t", "threshold": "x"}); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("MissingTask", func(t *testing.T) {
		e := &TaskCompletionEvaluator{Client: &stubProvider{}}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{}); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ProviderErrorAndNilResp", func(t *testing.T) {
		e := &TaskCompletionEvaluator{Client: &stubProvider{fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			_ = ctx
			_ = req
			return nil, errors.New("boom")
		}}}
		_, err := e.Evaluate(context.Background(), "x", map[string]any{"task": "t"})
		if err == nil || !strings.Contains(err.Error(), "llm") {
			t.Fatalf("err=%v", err)
		}

		e.Client = &stubProvider{}
		_, err = e.Evaluate(context.Background(), "x", map[string]any{"task": "t"})
		if err == nil || !strings.Contains(err.Error(), "nil llm response") {
			t.Fatalf("err=%v", err)
		}
	})

	t.Run("InvalidJSON", func(t *testing.T) {
		e := &TaskCompletionEvaluator{Client: &stubProvider{fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			_ = ctx
			_ = req
			return textResponse("not json"), nil
		}}}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{"task": "t"})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 {
			t.Fatalf("res=%#v", res)
		}
	})

	t.Run("OK_ClampAndThreshold", func(t *testing.T) {
		e := &TaskCompletionEvaluator{Client: &stubProvider{fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			_ = ctx
			_ = req
			return textResponse(`{"score": 2, "reasoning": "", "missing": ["a"]}`), nil
		}}}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{
			"task":      "t",
			"threshold": 2,
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed || res.Score != 1 {
			t.Fatalf("res=%#v", res)
		}
		if res.Message != "no reasoning provided" {
			t.Fatalf("msg=%q", res.Message)
		}
		if res.Details["threshold"].(float64) != 1 {
			t.Fatalf("Details=%#v", res.Details)
		}
	})

	t.Run("DefaultThreshold", func(t *testing.T) {
		e := &TaskCompletionEvaluator{Client: &stubProvider{fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			_ = ctx
			_ = req
			return textResponse(`{"score": 0.7, "reasoning": "ok", "missing": []}`), nil
		}}}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{
			"task":      "t",
			"threshold": 0,
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed {
			t.Fatalf("res=%#v", res)
		}
		if res.Details["threshold"].(float64) != 0.6 {
			t.Fatalf("Details=%#v", res.Details)
		}
	})

	t.Run("CriteriaIncludedInPrompt", func(t *testing.T) {
		gotPrompt := ""
		e := &TaskCompletionEvaluator{Client: &stubProvider{fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			_ = ctx
			if req == nil || len(req.Messages) != 1 {
				t.Fatalf("req=%#v", req)
			}
			gotPrompt = req.Messages[0].Content
			return textResponse(`{"score": 1, "reasoning": "ok", "missing": []}`), nil
		}}}

		res, err := e.Evaluate(context.Background(), "agent response", map[string]any{
			"task":     "t",
			"criteria": []any{"a", " ", "b"},
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed {
			t.Fatalf("res=%#v", res)
		}
		if !strings.Contains(gotPrompt, "## Completion Criteria") || !strings.Contains(gotPrompt, "- a") || !strings.Contains(gotPrompt, "- b") {
			t.Fatalf("prompt=%q", gotPrompt)
		}
		if strings.Contains(gotPrompt, "- \n") {
			t.Fatalf("unexpected empty criteria bullet in prompt=%q", gotPrompt)
		}
	})
}

func TestTaskCompletionEvaluator_ReturnType(t *testing.T) {
	e := &TaskCompletionEvaluator{Client: &stubProvider{fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
		_ = ctx
		_ = req
		return textResponse(`{"score": 1, "reasoning": "ok", "missing": []}`), nil
	}}}
	res, err := e.Evaluate(context.Background(), "x", map[string]any{"task": "t"})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if _, ok := any(res).(*evaluator.Result); !ok {
		t.Fatalf("type=%T", res)
	}
}
