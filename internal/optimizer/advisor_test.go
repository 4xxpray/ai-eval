package optimizer

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/runner"
)

func TestDiagnose_NilProvider(t *testing.T) {
	a := &Advisor{Provider: nil}
	_, err := a.Diagnose(context.Background(), &DiagnoseRequest{PromptContent: "x"})
	if err == nil || err.Error() != "advisor: nil provider" {
		t.Fatalf("expected nil provider error, got %v", err)
	}
}

func TestDiagnose_NilAdvisor(t *testing.T) {
	var a *Advisor
	_, err := a.Diagnose(context.Background(), &DiagnoseRequest{PromptContent: "x"})
	if err == nil {
		t.Fatal("expected error for nil advisor")
	}
}

func TestDiagnose_NilContext(t *testing.T) {
	a := &Advisor{Provider: &mockProvider{name: "test"}}
	//nolint:staticcheck // intentional nil context for test
	_, err := a.Diagnose(nil, &DiagnoseRequest{PromptContent: "x"})
	if err == nil || err.Error() != "advisor: nil context" {
		t.Fatalf("expected nil context error, got %v", err)
	}
}

func TestDiagnose_NilRequest(t *testing.T) {
	a := &Advisor{Provider: &mockProvider{name: "test"}}
	_, err := a.Diagnose(context.Background(), nil)
	if err == nil || err.Error() != "advisor: nil request" {
		t.Fatalf("expected nil request error, got %v", err)
	}
}

func TestDiagnose_EmptyPrompt(t *testing.T) {
	a := &Advisor{Provider: &mockProvider{name: "test"}}
	_, err := a.Diagnose(context.Background(), &DiagnoseRequest{PromptContent: "   "})
	if err == nil || err.Error() != "advisor: empty prompt_content" {
		t.Fatalf("expected empty prompt error, got %v", err)
	}
}

func TestDiagnose_ProviderError(t *testing.T) {
	a := &Advisor{Provider: &mockProvider{
		name: "test",
		err:  errors.New("rate limited"),
	}}
	_, err := a.Diagnose(context.Background(), &DiagnoseRequest{PromptContent: "Review code"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("expected wrapped error, got: %v", err)
	}
}

func TestDiagnose_NilResponse(t *testing.T) {
	a := &Advisor{Provider: &mockProvider{name: "test", response: nil}}
	_, err := a.Diagnose(context.Background(), &DiagnoseRequest{PromptContent: "Review code"})
	if err == nil {
		t.Fatal("expected error for nil response")
	}
}

func TestDiagnose_ValidResponse(t *testing.T) {
	resp := textResponse(`{
		"failure_patterns": ["missing_context", "output_format_unclear"],
		"root_causes": ["No output schema specified"],
		"suggestions": [
			{
				"id": "S1",
				"type": "specify_output_format",
				"description": "Add JSON output requirement",
				"before": "",
				"after": "Output valid JSON",
				"impact": "high",
				"priority": 1
			}
		]
	}`)
	a := &Advisor{Provider: &mockProvider{name: "test", response: resp}}
	result, err := a.Diagnose(context.Background(), &DiagnoseRequest{
		PromptContent: "Review code",
		EvalResults: []*runner.SuiteResult{{
			Suite:       "test_suite",
			PassRate:    0.5,
			PassedCases: 1,
			TotalCases:  2,
			AvgScore:    0.6,
			Results: []runner.RunResult{
				{CaseID: "c1", Passed: false, Score: 0.2, Trials: []runner.TrialResult{
					{Passed: false, Response: "bad output", Evaluations: []evaluator.Result{
						{Passed: false, Message: "regex mismatch"},
					}},
				}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.FailurePatterns) != 2 {
		t.Errorf("expected 2 patterns, got %d", len(result.FailurePatterns))
	}
	if len(result.RootCauses) != 1 {
		t.Errorf("expected 1 root cause, got %d", len(result.RootCauses))
	}
	if len(result.Suggestions) != 1 {
		t.Errorf("expected 1 suggestion, got %d", len(result.Suggestions))
	}
	if result.Suggestions[0].Priority != 1 {
		t.Errorf("expected priority 1, got %d", result.Suggestions[0].Priority)
	}
}

func TestDiagnose_MaxSuggestionsDefault(t *testing.T) {
	resp := textResponse(`{
		"failure_patterns": [],
		"root_causes": [],
		"suggestions": [
			{"id":"S1","type":"add_context","description":"d1","priority":3},
			{"id":"S2","type":"add_context","description":"d2","priority":1},
			{"id":"S3","type":"add_context","description":"d3","priority":2}
		]
	}`)
	a := &Advisor{Provider: &mockProvider{name: "test", response: resp}}
	result, err := a.Diagnose(context.Background(), &DiagnoseRequest{
		PromptContent:  "prompt",
		MaxSuggestions: 0, // should default to 5
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// All 3 should be included (under default limit of 5)
	if len(result.Suggestions) != 3 {
		t.Errorf("expected 3 suggestions, got %d", len(result.Suggestions))
	}
	// Should be sorted by priority
	if result.Suggestions[0].ID != "S2" {
		t.Errorf("expected S2 first (priority 1), got %s", result.Suggestions[0].ID)
	}
}

func TestDiagnose_MaxSuggestionsCapped(t *testing.T) {
	resp := textResponse(`{
		"failure_patterns": [],
		"root_causes": [],
		"suggestions": [
			{"id":"S1","type":"a","description":"d1","priority":1},
			{"id":"S2","type":"a","description":"d2","priority":2},
			{"id":"S3","type":"a","description":"d3","priority":3}
		]
	}`)
	a := &Advisor{Provider: &mockProvider{name: "test", response: resp}}
	result, err := a.Diagnose(context.Background(), &DiagnoseRequest{
		PromptContent:  "prompt",
		MaxSuggestions: 2,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Suggestions) != 2 {
		t.Errorf("expected 2 suggestions (capped), got %d", len(result.Suggestions))
	}
}

// --- Helper function tests ---

func TestNormalizePatternIDs(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		want []string
	}{
		{"empty", nil, nil},
		{"dedup_and_lowercase", []string{"A", "a", "B", " b "}, []string{"a", "b"}},
		{"trim_blanks", []string{"", " ", "x"}, []string{"x"}},
		{"sorted", []string{"z", "a", "m"}, []string{"a", "m", "z"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy to avoid modifying test data
			input := make([]string, len(tt.in))
			copy(input, tt.in)
			got := normalizePatternIDs(input)
			if len(got) != len(tt.want) {
				t.Fatalf("len: got %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("[%d] got %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestTrimStringSlice(t *testing.T) {
	input := []string{"  hello ", "", " world ", "  "}
	got := trimStringSlice(append([]string{}, input...))
	if len(got) != 2 {
		t.Fatalf("expected 2 items, got %d", len(got))
	}
	if got[0] != "hello" || got[1] != "world" {
		t.Errorf("unexpected: %v", got)
	}
}

func TestNormalizeSuggestions(t *testing.T) {
	input := []FixSuggestion{
		{ID: "S3", Type: "add", Description: "d3", Priority: 3},
		{ID: "", Type: "add", Description: "skip"},           // missing ID
		{ID: "S1", Type: "", Description: "skip"},            // missing type
		{ID: "S2", Type: "add", Description: ""},             // missing description
		{ID: "S1", Type: "add", Description: "d1", Priority: 0},  // priority defaults to 3
		{ID: "S4", Type: "add", Description: "d4", Priority: 10}, // priority capped to 5
	}
	got := normalizeSuggestions(append([]FixSuggestion{}, input...), 10)
	if len(got) != 3 {
		t.Fatalf("expected 3 valid suggestions, got %d", len(got))
	}
	// S1 (priority 3), S3 (priority 3), S4 (priority 5)
	if got[0].Priority != 3 || got[1].Priority != 3 || got[2].Priority != 5 {
		t.Errorf("unexpected priorities: %d, %d, %d", got[0].Priority, got[1].Priority, got[2].Priority)
	}
}

func TestNormalizeSuggestions_MaxLimit(t *testing.T) {
	input := []FixSuggestion{
		{ID: "S1", Type: "a", Description: "d1", Priority: 1},
		{ID: "S2", Type: "a", Description: "d2", Priority: 2},
		{ID: "S3", Type: "a", Description: "d3", Priority: 3},
	}
	got := normalizeSuggestions(append([]FixSuggestion{}, input...), 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 suggestions, got %d", len(got))
	}
}

func TestFormatPatternRules(t *testing.T) {
	got := formatPatternRules(nil)
	if got != "(none)" {
		t.Errorf("expected '(none)' for nil, got %q", got)
	}

	rules := []PatternRule{
		{ID: "missing_context", Title: "Missing context", Description: "No context provided"},
		{ID: "", Title: "skip", Description: "empty id"},
		{ID: "ambiguous", Title: "", Description: ""},
	}
	got = formatPatternRules(rules)
	if !strings.Contains(got, "missing_context") {
		t.Error("expected missing_context in output")
	}
	if strings.Contains(got, "skip") {
		t.Error("should skip empty ID rule")
	}
	if !strings.Contains(got, "ambiguous") {
		t.Error("expected ambiguous in output (uses ID as title fallback)")
	}
}

func TestFormatEvalResultsForDiagnosis_Empty(t *testing.T) {
	got := formatEvalResultsForDiagnosis(nil)
	if got != "No evaluation results available." {
		t.Errorf("unexpected: %s", got)
	}
	got = formatEvalResultsForDiagnosis([]*runner.SuiteResult{nil})
	if got != "No evaluation results available." {
		t.Errorf("unexpected for nil element: %s", got)
	}
}

func TestFormatEvalResultsForDiagnosis_WithData(t *testing.T) {
	results := []*runner.SuiteResult{{
		Suite:       "suite1",
		PassRate:    0.5,
		PassedCases: 1,
		TotalCases:  2,
		AvgScore:    0.6,
		Results: []runner.RunResult{
			{CaseID: "pass_case", Passed: true},
			{CaseID: "fail_case", Passed: false, Score: 0.2, PassAtK: 0.0,
				Trials: []runner.TrialResult{{
					Passed:   false,
					Response: "wrong answer",
					Evaluations: []evaluator.Result{
						{Passed: false, Message: "expected X got Y"},
						{Passed: true, Message: "ok"},
					},
				}},
			},
		},
	}}
	got := formatEvalResultsForDiagnosis(results)
	if !strings.Contains(got, "suite1") {
		t.Error("expected suite name")
	}
	if !strings.Contains(got, "fail_case") {
		t.Error("expected failed case")
	}
	if !strings.Contains(got, "expected X got Y") {
		t.Error("expected failure message")
	}
	if strings.Contains(got, "pass_case") {
		t.Error("should not include passing cases")
	}
}

func TestFormatEvalResultsForDiagnosis_AllPassed(t *testing.T) {
	results := []*runner.SuiteResult{{
		Suite:       "suite1",
		PassRate:    1.0,
		PassedCases: 1,
		TotalCases:  1,
		AvgScore:    1.0,
		Results: []runner.RunResult{
			{CaseID: "c1", Passed: true},
		},
	}}
	got := formatEvalResultsForDiagnosis(results)
	if !strings.Contains(got, "All cases passed") {
		t.Error("expected 'All cases passed' message")
	}
}

func TestTruncate(t *testing.T) {
	if truncate("hello", 10) != "hello" {
		t.Error("should not truncate short string")
	}
	if truncate("hello world", 5) != "hello...(truncated)" {
		t.Errorf("unexpected: %s", truncate("hello world", 5))
	}
	if truncate("anything", 0) != "" {
		t.Error("max=0 should return empty")
	}
}

func TestIndentBlock(t *testing.T) {
	got := indentBlock("line1\nline2\nline3", "  ")
	if got != "  line1\n  line2\n  line3" {
		t.Errorf("unexpected: %q", got)
	}
	// CRLF normalization
	got = indentBlock("a\r\nb", "> ")
	if got != "> a\n> b" {
		t.Errorf("unexpected CRLF handling: %q", got)
	}
}
