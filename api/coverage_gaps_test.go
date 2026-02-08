package api

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestHandlers_DeletePrompt_MissingName(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodDelete, "/api/prompts/%20", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_DeletePrompt_InvalidName(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodDelete, "/api/prompts/.hidden", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetPrompt_Returns500WhenDirMissing(t *testing.T) {
	dir := t.TempDir()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(cwd) })

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/example", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetPrompt_Success(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/example", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandlers_GetTestSuite_Returns500WhenDirMissing(t *testing.T) {
	dir := t.TempDir()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(cwd) })

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/tests/basic", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_StartRun_Returns500WhenPromptsDirMissing(t *testing.T) {
	dir := t.TempDir()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(cwd) })

	s := &Server{
		store: &fakeStore{},
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.6, Concurrency: 1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"all":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_StartRun_Returns500WhenTestsDirMissing(t *testing.T) {
	dir := t.TempDir()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(cwd) })

	if err := os.MkdirAll(promptsDir, 0o755); err != nil {
		t.Fatalf("MkdirAll prompts: %v", err)
	}
	promptPayload := []byte("name: example\nversion: v1\ndescription: example\ntemplate: hello\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "example.yaml"), promptPayload, 0o644); err != nil {
		t.Fatalf("WriteFile prompt: %v", err)
	}

	s := &Server{
		store: &fakeStore{},
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.6, Concurrency: 1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"all":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_StartRun_OverrideRequestParams(t *testing.T) {
	setupAPITestWorkspace(t)

	var cfg map[string]any
	st := &fakeStore{
		SaveRunFunc: func(ctx context.Context, run *store.RunRecord) error {
			cfg = run.Config
			return nil
		},
		SaveSuiteResultFunc: func(ctx context.Context, result *store.SuiteRecord) error {
			return nil
		},
	}

	s := &Server{
		store: st,
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.1, Concurrency: 1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"all":true,"trials":2,"threshold":0.7,"concurrency":3}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusCreated)
	}
	if cfg == nil || cfg["trials"] != 2 {
		t.Fatalf("config.trials: got %v", cfg["trials"])
	}
	if cfg["threshold"] != float64(0.7) {
		t.Fatalf("config.threshold: got %v", cfg["threshold"])
	}
	if cfg["concurrency"] != 3 {
		t.Fatalf("config.concurrency: got %v", cfg["concurrency"])
	}
	if cfg["all"] != true {
		t.Fatalf("config.all: got %v", cfg["all"])
	}
}

func TestHandlers_StartRun_TwoSuitesForPrompt(t *testing.T) {
	setupAPITestWorkspace(t)

	suitePayload := []byte("suite: zzz\nprompt: example\ncases:\n  - id: c2\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n")
	if err := os.WriteFile(filepath.Join(testsDir, "zzz.yaml"), suitePayload, 0o644); err != nil {
		t.Fatalf("WriteFile suite: %v", err)
	}

	suiteSaves := 0
	st := &fakeStore{
		SaveRunFunc: func(ctx context.Context, run *store.RunRecord) error {
			return nil
		},
		SaveSuiteResultFunc: func(ctx context.Context, result *store.SuiteRecord) error {
			suiteSaves++
			return nil
		},
	}

	s := &Server{
		store: st,
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.6, Concurrency: 1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"example"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusCreated)
	}
	if suiteSaves != 2 {
		t.Fatalf("SaveSuiteResult calls: got %d want %d", suiteSaves, 2)
	}
}

func TestHandlers_StartRun_NoSuitesForPrompt(t *testing.T) {
	setupAPITestWorkspace(t)

	payload := []byte("name: lonely\nversion: v1\ndescription: lonely\ntemplate: hi\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "lonely.yaml"), payload, 0o644); err != nil {
		t.Fatalf("WriteFile prompt: %v", err)
	}

	s := &Server{
		store: &fakeStore{
			SaveRunFunc: func(ctx context.Context, run *store.RunRecord) error { return nil },
			SaveSuiteResultFunc: func(ctx context.Context, result *store.SuiteRecord) error {
				return nil
			},
		},
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.6, Concurrency: 1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"lonely"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestCompactPrompts_EmptySlice(t *testing.T) {
	var nilSlice []*prompt.Prompt
	if got := compactPrompts(nilSlice); got != nil {
		t.Fatalf("expected nil")
	}
	empty := []*prompt.Prompt{}
	if got := compactPrompts(empty); got == nil || len(got) != 0 {
		t.Fatalf("expected empty slice")
	}
}

func TestFindPromptByName_SuccessAndNilSkip(t *testing.T) {
	ps := []*prompt.Prompt{nil, {Name: " a "}}
	got, err := findPromptByName(ps, "a")
	if err != nil {
		t.Fatalf("findPromptByName: %v", err)
	}
	if got == nil || got.Name != " a " {
		t.Fatalf("unexpected prompt: %#v", got)
	}
}

func TestFindSuiteByName_NilSkip(t *testing.T) {
	suites := []*testcase.TestSuite{nil, {Suite: " a "}}
	got := findSuiteByName(suites, "a")
	if got == nil || got.Suite != " a " {
		t.Fatalf("unexpected suite: %#v", got)
	}
}
