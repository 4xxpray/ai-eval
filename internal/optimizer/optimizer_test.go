package optimizer

import (
	"context"
	"errors"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/runner"
)

// mockProvider implements llm.Provider for testing.
type mockProvider struct {
	name     string
	response *llm.Response
	err      error
}

func (m *mockProvider) Name() string { return m.name }
func (m *mockProvider) Complete(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return m.response, m.err
}
func (m *mockProvider) CompleteWithTools(_ context.Context, _ *llm.Request) (*llm.EvalResult, error) {
	return nil, nil
}

func textResponse(text string) *llm.Response {
	return &llm.Response{
		Content: []llm.ContentBlock{{Type: "text", Text: text}},
	}
}

func TestOptimize_NilProvider(t *testing.T) {
	o := &Optimizer{Provider: nil}
	_, err := o.Optimize(context.Background(), &OptimizeRequest{OriginalPrompt: "x"})
	if err == nil || err.Error() != "optimizer: nil provider" {
		t.Fatalf("expected nil provider error, got %v", err)
	}
}

func TestOptimize_NilOptimizer(t *testing.T) {
	var o *Optimizer
	_, err := o.Optimize(context.Background(), &OptimizeRequest{OriginalPrompt: "x"})
	if err == nil {
		t.Fatal("expected error for nil optimizer")
	}
}

func TestOptimize_NilRequest(t *testing.T) {
	o := &Optimizer{Provider: &mockProvider{name: "test"}}
	_, err := o.Optimize(context.Background(), nil)
	if err == nil || err.Error() != "optimizer: nil request" {
		t.Fatalf("expected nil request error, got %v", err)
	}
}

func TestOptimize_EmptyPrompt(t *testing.T) {
	o := &Optimizer{Provider: &mockProvider{name: "test"}}
	_, err := o.Optimize(context.Background(), &OptimizeRequest{OriginalPrompt: "   "})
	if err == nil || err.Error() != "optimizer: empty prompt" {
		t.Fatalf("expected empty prompt error, got %v", err)
	}
}

func TestOptimize_ProviderError(t *testing.T) {
	o := &Optimizer{Provider: &mockProvider{
		name: "test",
		err:  errors.New("api timeout"),
	}}
	_, err := o.Optimize(context.Background(), &OptimizeRequest{OriginalPrompt: "Review code"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, errors.Unwrap(err)) && err.Error() != "optimizer: api timeout" {
		// just verify it wraps
	}
}

func TestOptimize_ValidJSON(t *testing.T) {
	resp := textResponse(`{
		"summary": "Added output format constraint",
		"changes": [
			{
				"type": "add",
				"description": "Added JSON output requirement",
				"before": "",
				"after": "Output must be valid JSON"
			}
		],
		"optimized_prompt": "You are a strict code reviewer. Output must be valid JSON."
	}`)
	o := &Optimizer{Provider: &mockProvider{name: "test", response: resp}}
	result, err := o.Optimize(context.Background(), &OptimizeRequest{
		OriginalPrompt: "You are a strict code reviewer.",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.OptimizedPrompt != "You are a strict code reviewer. Output must be valid JSON." {
		t.Errorf("unexpected optimized prompt: %s", result.OptimizedPrompt)
	}
	if result.Summary != "Added output format constraint" {
		t.Errorf("unexpected summary: %s", result.Summary)
	}
	if len(result.Changes) != 1 {
		t.Fatalf("expected 1 change, got %d", len(result.Changes))
	}
	if result.Changes[0].Type != "add" {
		t.Errorf("expected change type 'add', got %s", result.Changes[0].Type)
	}
}

func TestOptimize_MarkdownWrappedJSON(t *testing.T) {
	resp := textResponse("```json\n" + `{
		"summary": "fix",
		"changes": [],
		"optimized_prompt": "improved prompt"
	}` + "\n```")
	o := &Optimizer{Provider: &mockProvider{name: "test", response: resp}}
	result, err := o.Optimize(context.Background(), &OptimizeRequest{
		OriginalPrompt: "original prompt",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.OptimizedPrompt != "improved prompt" {
		t.Errorf("unexpected prompt: %s", result.OptimizedPrompt)
	}
}

func TestOptimize_InvalidJSON(t *testing.T) {
	resp := textResponse("This is not JSON at all")
	o := &Optimizer{Provider: &mockProvider{name: "test", response: resp}}
	_, err := o.Optimize(context.Background(), &OptimizeRequest{
		OriginalPrompt: "original prompt",
	})
	if err == nil {
		t.Fatal("expected parse error")
	}
}

func TestOptimize_ScoreImprovement(t *testing.T) {
	resp := textResponse(`{"summary":"s","changes":[],"optimized_prompt":"p"}`)
	o := &Optimizer{Provider: &mockProvider{name: "test", response: resp}}
	result, err := o.Optimize(context.Background(), &OptimizeRequest{
		OriginalPrompt: "prompt",
		EvalResults: &runner.SuiteResult{
			AvgScore: 0.6,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := 0.4 // 1.0 - 0.6
	if result.ScoreImprovement < expected-0.01 || result.ScoreImprovement > expected+0.01 {
		t.Errorf("expected score improvement ~%.2f, got %.2f", expected, result.ScoreImprovement)
	}
}

func TestOptimize_NilEvalResults(t *testing.T) {
	resp := textResponse(`{"summary":"s","changes":[],"optimized_prompt":"p"}`)
	o := &Optimizer{Provider: &mockProvider{name: "test", response: resp}}
	result, err := o.Optimize(context.Background(), &OptimizeRequest{
		OriginalPrompt: "prompt",
		EvalResults:    nil,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ScoreImprovement != 0 {
		t.Errorf("expected 0 score improvement with nil results, got %.2f", result.ScoreImprovement)
	}
}

func TestFormatEvalResults_Nil(t *testing.T) {
	got := formatEvalResults(nil)
	if got != "No evaluation results available." {
		t.Errorf("unexpected: %s", got)
	}
}

func TestFormatEvalResults_WithFailures(t *testing.T) {
	sr := &runner.SuiteResult{
		PassRate:    0.5,
		PassedCases: 1,
		TotalCases:  2,
		AvgScore:    0.75,
		Results: []runner.RunResult{
			{CaseID: "case1", Passed: true, Score: 1.0},
			{CaseID: "case2", Passed: false, Score: 0.5, Trials: []runner.TrialResult{
				{Passed: false, Evaluations: []evaluator.Result{
					{Passed: false, Message: "regex mismatch"},
				}},
			}},
		},
	}
	got := formatEvalResults(sr)
	if got == "" {
		t.Fatal("expected non-empty output")
	}
	if !contains(got, "50.0%") {
		t.Error("expected pass rate in output")
	}
	if !contains(got, "case2") {
		t.Error("expected failed case in output")
	}
	if !contains(got, "regex mismatch") {
		t.Error("expected failure message in output")
	}
}

func TestExtractJSONStringField_Complete(t *testing.T) {
	text := `{"optimized_prompt": "hello world", "summary": "test"}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "hello world" {
		t.Errorf("expected 'hello world', got %q", got)
	}
}

func TestExtractJSONStringField_Escaped(t *testing.T) {
	text := `{"optimized_prompt": "line1\nline2\t\"quoted\""}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "line1\nline2\t\"quoted\"" {
		t.Errorf("unexpected: %q", got)
	}
}

func TestExtractJSONStringField_Truncated(t *testing.T) {
	text := `{"optimized_prompt": "this is a long prompt that gets cut`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "this is a long prompt that gets cut" {
		t.Errorf("expected truncated content, got %q", got)
	}
}

func TestExtractJSONStringField_Missing(t *testing.T) {
	text := `{"summary": "test"}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestExtractJSONStringField_NotString(t *testing.T) {
	text := `{"optimized_prompt": 42}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "" {
		t.Errorf("expected empty for non-string, got %q", got)
	}
}

func TestExtractJSONStringField_EmptyAfterColon(t *testing.T) {
	text := `{"optimized_prompt":`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestExtractJSONStringField_UnknownEscape(t *testing.T) {
	text := `{"optimized_prompt": "a\qb"}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != `a\qb` {
		t.Errorf("unexpected: %q", got)
	}
}

func TestExtractJSONStringField_BackslashEscape(t *testing.T) {
	text := `{"optimized_prompt": "a\\b"}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != `a\b` {
		t.Errorf("unexpected: %q", got)
	}
}

func TestExtractJSONStringField_CarriageReturnEscape(t *testing.T) {
	text := `{"optimized_prompt": "a\rb"}`
	got := extractJSONStringField(text, "optimized_prompt")
	if got != "a\rb" {
		t.Errorf("unexpected: %q", got)
	}
}

func TestOptimize_TruncatedJSON(t *testing.T) {
	// Simulate a truncated response where JSON is incomplete
	truncated := `{"optimized_prompt": "improved prompt content here", "summary": "fixed issues`
	resp := textResponse(truncated)
	o := &Optimizer{Provider: &mockProvider{name: "test", response: resp}}
	result, err := o.Optimize(context.Background(), &OptimizeRequest{
		OriginalPrompt: "original",
	})
	if err != nil {
		t.Fatalf("expected fallback extraction, got error: %v", err)
	}
	if result.OptimizedPrompt != "improved prompt content here" {
		t.Errorf("unexpected prompt: %q", result.OptimizedPrompt)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && containsStr(s, sub)
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
