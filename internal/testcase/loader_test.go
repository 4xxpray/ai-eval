package testcase

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadFromFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "suite.yaml")

	const in = `
suite: example_suite
prompt: code_review
description: Example test suite
cases:
  - id: basic_contains
    input:
      language: go
      diff: |
        diff --git a/a.go b/a.go
        index 0000000..1111111 100644
        --- a/a.go
        +++ b/a.go
    expected:
      contains:
        - correctness
      not_contains:
        - LGTM
  - id: llm_judge_only
    input: {}
    expected: {}
    evaluators:
      - type: llm_judge
        criteria: Must be a strict review with actionable feedback.
        score_threshold: 0.8
    trials: 2
    max_steps: 7
    tool_mocks:
      - name: git
        response: "ok"
        match:
          cmd: "status"
`
	if err := os.WriteFile(path, []byte(in), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	s, err := LoadFromFile(path)
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	if s.Suite != "example_suite" {
		t.Fatalf("Suite: got %q want %q", s.Suite, "example_suite")
	}
	if s.Prompt != "code_review" {
		t.Fatalf("Prompt: got %q want %q", s.Prompt, "code_review")
	}
	if len(s.Cases) != 2 {
		t.Fatalf("len(Cases): got %d want %d", len(s.Cases), 2)
	}
	if got := s.Cases[1].Trials; got != 2 {
		t.Fatalf("Cases[1].Trials: got %d want %d", got, 2)
	}
	if got := s.Cases[1].MaxSteps; got != 7 {
		t.Fatalf("Cases[1].MaxSteps: got %d want %d", got, 7)
	}
	if len(s.Cases[1].ToolMocks) != 1 || s.Cases[1].ToolMocks[0].Name != "git" {
		t.Fatalf("Cases[1].ToolMocks: got %#v", s.Cases[1].ToolMocks)
	}
}

func TestLoadFromFile_Missing(t *testing.T) {
	t.Parallel()

	_, err := LoadFromFile(filepath.Join(t.TempDir(), "missing.yaml"))
	if err == nil {
		t.Fatalf("LoadFromFile: expected error")
	}
}

func TestLoadFromDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("WriteFile(%s): %v", name, err)
		}
	}

	write("b.yaml", "suite: b\nprompt: p\ncases:\n  - id: b1\n    input: {}\n    expected:\n      exact_match: ok\n")
	write("a.yml", "suite: a\nprompt: p\ncases:\n  - id: a1\n    input: {}\n    expected:\n      exact_match: ok\n")
	write("ignored.txt", "nope\n")

	ss, err := LoadFromDir(dir)
	if err != nil {
		t.Fatalf("LoadFromDir: %v", err)
	}
	if len(ss) != 2 {
		t.Fatalf("len: got %d want %d", len(ss), 2)
	}
	if ss[0].Suite != "a" || ss[1].Suite != "b" {
		t.Fatalf("order: got %q, %q", ss[0].Suite, ss[1].Suite)
	}
}

func TestLoadFromDir_BadYAML(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "bad.yaml"), []byte(":\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := LoadFromDir(dir)
	if err == nil {
		t.Fatalf("LoadFromDir: expected error")
	}
}

func TestLoadFromFile_BadYAML(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "bad.yaml")
	if err := os.WriteFile(path, []byte(":\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := LoadFromFile(path)
	if err == nil {
		t.Fatalf("LoadFromFile: expected error")
	}
}

func TestLoadFromFile_JSON(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "suite.yaml")

	const in = `{
  "suite": "s",
  "prompt": "p",
  "cases": [
    {
      "id": "c1",
      "input": {},
      "expected": { "exact_match": "ok" }
    }
  ]
}`
	if err := os.WriteFile(path, []byte(in), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	s, err := LoadFromFile(path)
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	if s.Suite != "s" {
		t.Fatalf("Suite: got %q want %q", s.Suite, "s")
	}
}

func TestLoadFromFile_InvalidSuite(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "suite.yaml")

	const in = `
suite: s
prompt: p
cases:
  - id: c1
    input: {}
    expected: {}
`
	if err := os.WriteFile(path, []byte(in), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := LoadFromFile(path)
	if err == nil {
		t.Fatalf("LoadFromFile: expected error")
	}
	if !strings.Contains(err.Error(), "validate") {
		t.Fatalf("LoadFromFile: got %v", err)
	}
}

func TestLoadFromDir_Missing(t *testing.T) {
	t.Parallel()

	_, err := LoadFromDir(filepath.Join(t.TempDir(), "missing"))
	if err == nil {
		t.Fatalf("LoadFromDir: expected error")
	}
}

func TestValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		suite     *TestSuite
		wantError string
	}{
		{
			name:      "nil",
			suite:     nil,
			wantError: "nil suite",
		},
		{
			name:      "missing suite name",
			suite:     &TestSuite{Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}}}},
			wantError: "missing suite name",
		},
		{
			name:      "duplicate case id",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}}, {ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}}}},
			wantError: "duplicate id",
		},
		{
			name:      "missing prompt reference",
			suite:     &TestSuite{Suite: "s", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}}}},
			wantError: "missing prompt reference",
		},
		{
			name:      "no cases",
			suite:     &TestSuite{Suite: "s", Prompt: "p"},
			wantError: "no cases",
		},
		{
			name:      "missing case id",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}}}},
			wantError: "missing id",
		},
		{
			name:      "missing input",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: nil, Expected: Expected{ExactMatch: "ok"}}}},
			wantError: "missing input",
		},
		{
			name:      "negative trials",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}, Trials: -1}}},
			wantError: "trials must be >= 0",
		},
		{
			name:      "negative max steps",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}, MaxSteps: -1}}},
			wantError: "max_steps must be >= 0",
		},
		{
			name:      "tool mock missing name",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ExactMatch: "ok"}, ToolMocks: []ToolMock{{Name: " ", Response: "ok"}}}}},
			wantError: "tool_mocks[0]: missing name",
		},
		{
			name:      "no assertions",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{}}}},
			wantError: "no expected assertions",
		},
		{
			name:      "expected contains empty",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{Contains: []string{" "}}}}},
			wantError: "expected.contains[0]: empty string",
		},
		{
			name:      "expected not_contains empty",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{NotContains: []string{" "}}}}},
			wantError: "expected.not_contains[0]: empty string",
		},
		{
			name:      "bad regex",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{Regex: []string{"["}}}}},
			wantError: "expected.regex",
		},
		{
			name:      "expected tool call missing name",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ToolCalls: []ToolCallExpect{{Name: " "}}}}}},
			wantError: "expected.tool_calls[0]: missing name",
		},
		{
			name:      "expected tool call negative order",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ToolCalls: []ToolCallExpect{{Name: "t1", Order: -1}}}}}},
			wantError: "order must be >= 0",
		},
		{
			name:      "expected tool call duplicate order",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{ToolCalls: []ToolCallExpect{{Name: "t1", Order: 1}, {Name: "t2", Order: 1}}}}}},
			wantError: "duplicate order 1",
		},
		{
			name:      "unknown evaluator",
			suite:     &TestSuite{Suite: "s", Prompt: "p", Cases: []TestCase{{ID: "c1", Input: map[string]any{}, Expected: Expected{}, Evaluators: []EvaluatorConfig{{Type: "nope"}}}}},
			wantError: "unknown type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := Validate(tt.suite)
			if err == nil {
				t.Fatalf("Validate: expected error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Validate: got %v want substring %q", err, tt.wantError)
			}
		})
	}
}

func TestValidate_NewEvaluatorTypes(t *testing.T) {
	t.Parallel()

	suite := &TestSuite{
		Suite:  "s",
		Prompt: "p",
		Cases: []TestCase{{
			ID:       "c1",
			Input:    map[string]any{},
			Expected: Expected{},
			Evaluators: []EvaluatorConfig{
				{Type: "faithfulness", Context: "ctx", ScoreThreshold: 0.8},
				{Type: "relevancy", Question: "q", ScoreThreshold: 0.8},
				{Type: "precision", Context: "ctx", Question: "q"},
				{Type: "task_completion", Task: "do it", CriteriaList: []string{"a", "b"}},
				{Type: "tool_selection", ExpectedTools: []string{"search"}},
				{Type: "efficiency", MaxSteps: 5, MaxTokens: 1000},
				{Type: "hallucination", GroundTruth: "gt", ScoreThreshold: 0.9},
				{Type: "toxicity", ScoreThreshold: 0.1},
				{Type: "bias", Categories: []string{"gender"}, ScoreThreshold: 0.1},
				{Type: "tool_call"},
				{Type: "json_schema"},
				{Type: "contains"},
				{Type: "regex"},
				{Type: "exact"},
			},
		}},
	}

	if err := Validate(suite); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestValidate_EvaluatorErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		e         EvaluatorConfig
		wantError string
	}{
		{name: "missing type", e: EvaluatorConfig{Type: "  "}, wantError: "missing type"},
		{name: "negative score threshold", e: EvaluatorConfig{Type: "regex", ScoreThreshold: -0.1}, wantError: "score_threshold must be >= 0"},
		{name: "llm_judge missing criteria", e: EvaluatorConfig{Type: "llm_judge"}, wantError: "missing criteria"},
		{name: "llm_judge negative score scale", e: EvaluatorConfig{Type: "llm_judge", Criteria: "c", ScoreScale: -1}, wantError: "score_scale must be >= 0"},
		{name: "similarity missing reference", e: EvaluatorConfig{Type: "similarity"}, wantError: "missing reference"},
		{name: "factuality missing ground_truth", e: EvaluatorConfig{Type: "factuality"}, wantError: "missing ground_truth"},
		{name: "faithfulness missing context", e: EvaluatorConfig{Type: "faithfulness"}, wantError: "missing context"},
		{name: "relevancy missing question", e: EvaluatorConfig{Type: "relevancy"}, wantError: "missing question"},
		{name: "precision missing context", e: EvaluatorConfig{Type: "precision", Question: "q"}, wantError: "missing context"},
		{name: "precision missing question", e: EvaluatorConfig{Type: "precision", Context: "c"}, wantError: "missing question"},
		{name: "task_completion missing task", e: EvaluatorConfig{Type: "task_completion"}, wantError: "missing task"},
		{name: "efficiency negative max_steps", e: EvaluatorConfig{Type: "efficiency", MaxSteps: -1}, wantError: "max_steps must be >= 0"},
		{name: "efficiency negative max_tokens", e: EvaluatorConfig{Type: "efficiency", MaxTokens: -1}, wantError: "max_tokens must be >= 0"},
		{name: "hallucination missing ground_truth", e: EvaluatorConfig{Type: "hallucination"}, wantError: "missing ground_truth"},
		{name: "bias empty category", e: EvaluatorConfig{Type: "bias", Categories: []string{" "}}, wantError: "categories[0]: empty string"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			suite := &TestSuite{
				Suite:  "s",
				Prompt: "p",
				Cases: []TestCase{{
					ID:         "c1",
					Input:      map[string]any{},
					Expected:   Expected{},
					Evaluators: []EvaluatorConfig{tt.e},
				}},
			}

			err := Validate(suite)
			if err == nil {
				t.Fatalf("Validate: expected error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Validate: got %v want substring %q", err, tt.wantError)
			}
		})
	}
}
