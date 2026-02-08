package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

func TestHandleDiagnose_TwoSuitesSamePrompt(t *testing.T) {
	p := &fakeProvider{
		CompleteFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{
  "failure_patterns": [],
  "root_causes": [],
  "suggestions": [{
    "id": "S1",
    "type": "rewrite_prompt",
    "description": "d",
    "before": "b",
    "after": "a",
    "impact": "high",
    "priority": 1
  }]
}`}}}, nil
		},
		CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
			return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
		},
	}

	s := &Server{provider: p, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	testsYAML := "suite: a\nprompt: p\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n---\nsuite: b\nprompt: p\ncases:\n  - id: c2\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n"
	body, err := json.Marshal(map[string]any{
		"prompt_content": "hello",
		"tests_yaml":     testsYAML,
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if out["suites"] == nil || out["diagnosis"] == nil {
		t.Fatalf("expected suites and diagnosis")
	}
}
