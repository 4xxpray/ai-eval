package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

func TestHandleOptimize_NotInitialized(t *testing.T) {
	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/optimize", bytes.NewBufferString(`{"prompt_content":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandleOptimize_BadJSON(t *testing.T) {
	s := &Server{provider: &fakeProvider{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/optimize", bytes.NewBufferString("{"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleOptimize_EmptyPromptContent(t *testing.T) {
	s := &Server{provider: &fakeProvider{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/optimize", bytes.NewBufferString(`{"prompt_content":"   "}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleOptimize_GeneratorError(t *testing.T) {
	p := &fakeProvider{
		CompleteFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{provider: p}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/optimize", bytes.NewBufferString(`{"prompt_content":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandleOptimize_OptimizerError(t *testing.T) {
	p := &fakeProvider{
		CompleteFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			if req == nil || len(req.Messages) == 0 {
				return nil, errors.New("missing message")
			}
			content := req.Messages[0].Content
			switch {
			case strings.Contains(content, "prompt evaluation expert"):
				return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{
  "analysis": "ok",
  "is_system_prompt": false,
  "suggestions": [],
  "test_cases": [{
    "id": "c1",
    "description": "d",
    "input": {"text":"hi"},
    "expected": {"contains": [], "not_contains": [], "regex": []},
    "evaluators": [{"type":"efficiency","criteria":"","score_threshold":0}]
  }]
}`}}}, nil
			case strings.Contains(content, "prompt engineering expert"):
				return nil, errors.New("boom")
			default:
				return nil, errors.New("unexpected prompt")
			}
		},
		CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
			return &llm.EvalResult{TextContent: "x", InputTokens: 0, OutputTokens: 0}, nil
		},
	}

	s := &Server{provider: p}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/optimize", bytes.NewBufferString(`{"prompt_content":"hello"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandleDiagnose_NotInitialized(t *testing.T) {
	s := &Server{provider: &fakeProvider{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString(`{"prompt_content":"x","tests_yaml":"y"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandleDiagnose_BadJSON(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString("{"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_EmptyPromptContent(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString(`{"prompt_content":"   ","tests_yaml":"suite: s\\nprompt: p\\ncases:\\n  - id: c1\\n    input: {text: hi}\\n    expected: {contains: [ok]}\\n"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_EmptyTestsYAML(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString(`{"prompt_content":"x","tests_yaml":"   "}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_InvalidTestsYAML(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString(`{"prompt_content":"x","tests_yaml":"suite: ["}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_NoSuitesProvided(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString(`{"prompt_content":"x","tests_yaml":"# comment\\n"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_MixedSystemPromptAcrossSuites(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	testsYAML := strings.Join([]string{
		"suite: a\nprompt: p\nis_system_prompt: true\ncases:\n  - id: c1\n    input: {text: hi}\n    expected: {contains: [ok]}\n",
		"---\n",
		"suite: b\nprompt: p\nis_system_prompt: false\ncases:\n  - id: c2\n    input: {text: hi}\n    expected: {contains: [ok]}\n",
	}, "")
	body := `{"prompt_content":"x","tests_yaml":` + strconv.Quote(testsYAML) + `}`

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_InvalidThreshold(t *testing.T) {
	s := &Server{provider: &fakeProvider{}, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 2}}}
	r := newTestRouterForServer(t, s)

	testsYAML := "suite: s\nprompt: p\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n"
	body, err := json.Marshal(map[string]any{
		"prompt_content": "x",
		"tests_yaml":     testsYAML,
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandleDiagnose_AdvisorError(t *testing.T) {
	p := &fakeProvider{
		CompleteFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			return nil, errors.New("boom")
		},
		CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
			return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
		},
	}

	s := &Server{provider: p, config: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}}}
	r := newTestRouterForServer(t, s)

	testsYAML := "suite: s\nprompt: p\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n"
	body, err := json.Marshal(map[string]any{
		"prompt_content": "x",
		"tests_yaml":     testsYAML,
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}
