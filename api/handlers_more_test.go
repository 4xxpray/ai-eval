package api

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func newTestRouterForServer(t *testing.T, s *Server) *gin.Engine {
	t.Helper()

	gin.SetMode(gin.TestMode)
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	if s.router == nil {
		s.router = gin.New()
	}
	if err := s.registerRoutes(); err != nil {
		t.Fatalf("registerRoutes: %v", err)
	}
	return s.router
}

func TestHandlers_ListPrompts_FilterByName(t *testing.T) {
	setupAPITestWorkspace(t)

	other := []byte("name: other\nversion: v1\ndescription: other prompt\ntemplate: hi\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "other.yaml"), other, 0o644); err != nil {
		t.Fatalf("WriteFile other prompt: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts?name=EXAMPLE", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out []map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len(prompts): got %d want %d", len(out), 1)
	}
	if got := strings.ToLower(strings.TrimSpace(out[0]["Name"].(string))); got != "example" {
		t.Fatalf("prompt[0].Name: got %q want %q", got, "example")
	}
}

func TestHandlers_ListPrompts_Returns500WhenDirMissing(t *testing.T) {
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

	req := httptest.NewRequest(http.MethodGet, "/api/prompts", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetPrompt_MissingName(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/%20", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetPrompt_NotFound(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/nope", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_GetPrompt_ConflictDuplicateName(t *testing.T) {
	setupAPITestWorkspace(t)

	dup := []byte("name: example\nversion: v2\ndescription: dup\ntemplate: hi\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "dup.yaml"), dup, 0o644); err != nil {
		t.Fatalf("WriteFile dup prompt: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/example", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusConflict {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusConflict)
	}
}

func TestHandlers_UpsertPrompt_BadJSON(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodPost, "/api/prompts", bytes.NewBufferString("{"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_UpsertPrompt_MissingName(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodPost, "/api/prompts", bytes.NewBufferString(`{"template":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_UpsertPrompt_InvalidName(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodPost, "/api/prompts", bytes.NewBufferString(`{"name":".hidden","template":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_UpsertPrompt_MkdirAllError(t *testing.T) {
	setupAPITestWorkspace(t)

	if err := os.RemoveAll(promptsDir); err != nil {
		t.Fatalf("RemoveAll prompts: %v", err)
	}
	if err := os.WriteFile(promptsDir, []byte("not a dir"), 0o644); err != nil {
		t.Fatalf("WriteFile prompts: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodPost, "/api/prompts", bytes.NewBufferString(`{"name":"x","template":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_UpsertPrompt_WriteFileError(t *testing.T) {
	setupAPITestWorkspace(t)

	if err := os.MkdirAll(filepath.Join(promptsDir, "bad.yaml"), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodPost, "/api/prompts", bytes.NewBufferString(`{"name":"bad","template":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_UpsertPrompt_Success(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodPost, "/api/prompts", bytes.NewBufferString(`{"name":"new","version":"v1","template":"hello"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	if _, err := os.Stat(filepath.Join(promptsDir, "new.yaml")); err != nil {
		t.Fatalf("Stat new.yaml: %v", err)
	}
}

func TestHandlers_DeletePrompt_NotFound(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodDelete, "/api/prompts/nope", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_DeletePrompt_InternalError(t *testing.T) {
	setupAPITestWorkspace(t)

	path := filepath.Join(promptsDir, "bad.yaml")
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	if err := os.WriteFile(filepath.Join(path, "x"), []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodDelete, "/api/prompts/bad", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_DeletePrompt_Success(t *testing.T) {
	setupAPITestWorkspace(t)

	payload := []byte("name: del\nversion: v1\ndescription: del\ntemplate: hi\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "del.yaml"), payload, 0o644); err != nil {
		t.Fatalf("WriteFile del prompt: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodDelete, "/api/prompts/del", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNoContent)
	}
	if _, err := os.Stat(filepath.Join(promptsDir, "del.yaml")); !os.IsNotExist(err) {
		t.Fatalf("expected del.yaml removed, got %v", err)
	}
}

func TestHandlers_ListTests_FilterByPrompt(t *testing.T) {
	setupAPITestWorkspace(t)

	suitePayload := []byte("suite: other\nprompt: other\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n")
	if err := os.WriteFile(filepath.Join(testsDir, "other.yaml"), suitePayload, 0o644); err != nil {
		t.Fatalf("WriteFile suite: %v", err)
	}

	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/tests?prompt=example", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out []map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len(suites): got %d want %d", len(out), 1)
	}
	if out[0]["Suite"] != "basic" {
		t.Fatalf("suite[0].Suite: got %v want %q", out[0]["Suite"], "basic")
	}
}

func TestHandlers_ListTests_Returns500WhenDirMissing(t *testing.T) {
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

	req := httptest.NewRequest(http.MethodGet, "/api/tests", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetTestSuite_MissingName(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/tests/%20", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetTestSuite_NotFound(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/tests/nope", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_GetTestSuite_Success(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/tests/basic", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandlers_StartRun_NotInitialized(t *testing.T) {
	setupAPITestWorkspace(t)

	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"example"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_StartRun_BadJSON(t *testing.T) {
	setupAPITestWorkspace(t)

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

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString("{"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_MutuallyExclusive(t *testing.T) {
	setupAPITestWorkspace(t)

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

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"example","all":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_MissingPromptAndAll(t *testing.T) {
	setupAPITestWorkspace(t)

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

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_InvalidTrials(t *testing.T) {
	setupAPITestWorkspace(t)

	s := &Server{
		store: &fakeStore{},
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 0, Threshold: 0.6, Concurrency: 1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"example"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_InvalidThreshold(t *testing.T) {
	setupAPITestWorkspace(t)

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

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"example","threshold":1.1}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_UnknownPrompt(t *testing.T) {
	setupAPITestWorkspace(t)

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

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"nope"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_StartRun_NoSuitesFound(t *testing.T) {
	setupAPITestWorkspace(t)

	if err := os.Remove(filepath.Join(testsDir, "basic.yaml")); err != nil {
		t.Fatalf("Remove basic.yaml: %v", err)
	}

	s := &Server{
		store: &fakeStore{},
		provider: &fakeProvider{
			CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
				return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
			},
		},
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.6, Concurrency: 0}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"all":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_IndexPromptsError(t *testing.T) {
	setupAPITestWorkspace(t)

	payload := []byte("version: v1\ndescription: bad\ntemplate: hi\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "bad.yaml"), payload, 0o644); err != nil {
		t.Fatalf("WriteFile bad prompt: %v", err)
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

func TestHandlers_StartRun_IndexSuitesByPromptError(t *testing.T) {
	setupAPITestWorkspace(t)

	payload := []byte("suite: ghost\nprompt: ghost\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n")
	if err := os.WriteFile(filepath.Join(testsDir, "basic.yaml"), payload, 0o644); err != nil {
		t.Fatalf("WriteFile suite: %v", err)
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

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_StartRun_SaveRunError(t *testing.T) {
	setupAPITestWorkspace(t)

	saveErr := errors.New("save failed")
	st := &fakeStore{
		SaveRunFunc: func(ctx context.Context, run *store.RunRecord) error {
			return saveErr
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

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_StartRun_Success(t *testing.T) {
	setupAPITestWorkspace(t)

	var savedRun *store.RunRecord
	suiteSaves := 0

	st := &fakeStore{
		SaveRunFunc: func(ctx context.Context, run *store.RunRecord) error {
			savedRun = run
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
		config: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.6, Concurrency: -1}},
	}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewBufferString(`{"prompt":"example"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusCreated)
	}
	if savedRun == nil || strings.TrimSpace(savedRun.ID) == "" {
		t.Fatalf("expected SaveRun called with run ID")
	}
	if suiteSaves == 0 {
		t.Fatalf("expected SaveSuiteResult called")
	}

	var body struct {
		Run     *store.RunRecord `json:"run"`
		Summary map[string]any   `json:"summary"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if body.Run == nil || strings.TrimSpace(body.Run.ID) == "" {
		t.Fatalf("expected response run.id")
	}
	if got := int(body.Summary["total_suites"].(float64)); got != 1 {
		t.Fatalf("summary.total_suites: got %d want %d", got, 1)
	}
}

func TestHandlers_ListRuns_InvalidLimit(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs?limit=wat", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_ListRuns_InvalidSince(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs?since=wat", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_ListRuns_Success(t *testing.T) {
	wantSince := time.Date(2025, 2, 3, 0, 0, 0, 0, time.UTC)
	wantUntil := time.Date(2025, 2, 4, 0, 0, 0, 0, time.UTC)

	st := &fakeStore{
		ListRunsFunc: func(ctx context.Context, filter store.RunFilter) ([]*store.RunRecord, error) {
			if filter.PromptName != "p" || filter.PromptVersion != "v" || filter.Limit != 7 {
				return nil, fmt.Errorf("unexpected filter: %+v", filter)
			}
			if !filter.Since.Equal(wantSince) || !filter.Until.Equal(wantUntil) {
				return nil, fmt.Errorf("unexpected since/until: %v %v", filter.Since, filter.Until)
			}
			return []*store.RunRecord{{ID: "r1"}}, nil
		},
	}

	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs?prompt=p&version=v&limit=7&since=2025-02-03&until=2025-02-04", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandlers_GetRun_NotFound(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return nil, sql.ErrNoRows
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_GetRun_Success(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return &store.RunRecord{ID: id}, nil
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandlers_GetRunResults_Success(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return &store.RunRecord{ID: id}, nil
		},
		GetSuiteResultsFunc: func(ctx context.Context, runID string) ([]*store.SuiteRecord, error) {
			return []*store.SuiteRecord{{RunID: runID}}, nil
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1/results", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandlers_GetPromptHistory_Success(t *testing.T) {
	st := &fakeStore{
		GetPromptHistoryFunc: func(ctx context.Context, promptName string, limit int) ([]*store.SuiteRecord, error) {
			if promptName != "p" || limit != 3 {
				return nil, fmt.Errorf("unexpected args: %q %d", promptName, limit)
			}
			return []*store.SuiteRecord{{PromptName: promptName}}, nil
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/history/p?limit=3", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandlers_CompareVersions_NotFoundWhenNoRuns(t *testing.T) {
	st := &fakeStore{
		GetVersionComparisonFunc: func(ctx context.Context, promptName, v1, v2 string) (*store.VersionComparison, error) {
			return nil, errors.New("no runs")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/compare", bytes.NewBufferString(`{"prompt":"p","v1":"a","v2":"b"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_CompareVersions_Success(t *testing.T) {
	st := &fakeStore{
		GetVersionComparisonFunc: func(ctx context.Context, promptName, v1, v2 string) (*store.VersionComparison, error) {
			return &store.VersionComparison{PromptName: promptName, V1: v1, V2: v2}, nil
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/compare", bytes.NewBufferString(`{"prompt":"p","v1":"a","v2":"b"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestHandleOptimize_AlreadyGoodSkipsOptimizer(t *testing.T) {
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
  "suggestions": ["s1"],
  "test_cases": [{
    "id": "c1",
    "description": "d",
    "input": {"text":"hi"},
    "expected": {"contains": [], "not_contains": [], "regex": []},
    "evaluators": [{"type":"efficiency","criteria":"","score_threshold":0}]
  }]
}`}}}, nil
			default:
				return nil, fmt.Errorf("unexpected Complete content: %q", content)
			}
		},
		CompleteWithToolsFunc: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
			return &llm.EvalResult{TextContent: "x", InputTokens: 1, OutputTokens: 1}, nil
		},
	}

	s := &Server{provider: p}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/optimize", bytes.NewBufferString(`{"prompt_content":"hello","prompt_name":"p","num_cases":1}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	if !strings.Contains(rec.Body.String(), "No optimization needed") {
		t.Fatalf("expected summary to indicate no optimization needed")
	}
}

func TestHandleOptimize_UsesOptimizerOnLowScore(t *testing.T) {
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
				return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{
  "optimized_prompt": "better",
  "summary": "s",
  "changes": [{"type":"modify","description":"x"}]
}`}}}, nil
			default:
				return nil, fmt.Errorf("unexpected Complete content: %q", content)
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

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	var out map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if out["optimized_prompt"] != "better" {
		t.Fatalf("optimized_prompt: got %v want %q", out["optimized_prompt"], "better")
	}
}

func TestHandleDiagnose_Success(t *testing.T) {
	p := &fakeProvider{
		CompleteFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			if req == nil || len(req.Messages) == 0 {
				return nil, errors.New("missing message")
			}
			content := req.Messages[0].Content
			if !strings.Contains(content, "prompt debugging advisor") {
				return nil, fmt.Errorf("unexpected Complete content: %q", content)
			}
			return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{
  "failure_patterns": ["output_format_unclear"],
  "root_causes": ["r1"],
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

	cfg := &config.Config{Evaluation: config.EvaluationConfig{Trials: 0, Threshold: 0.6, Concurrency: 0}}
	s := &Server{provider: p, config: cfg}
	r := newTestRouterForServer(t, s)

	testsYAML := "suite: s1\nprompt: p1\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n"
	body := map[string]any{
		"prompt_content":  "hello",
		"tests_yaml":      testsYAML,
		"max_suggestions": 1,
	}
	buf, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/diagnose", bytes.NewReader(buf))
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
	if out["diagnosis"] == nil {
		t.Fatalf("expected diagnosis in response")
	}
}

func TestDecodeTestSuitesFromYAML_ValidMultiDoc(t *testing.T) {
	raw := strings.Join([]string{
		"suite: a\nprompt: p\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains: [\"ok\"]\n",
		"---\n",
		"suite: b\nprompt: p\ncases:\n  - id: c2\n    input:\n      text: hi\n    expected:\n      contains: [\"ok\"]\n",
	}, "")
	suites, err := decodeTestSuitesFromYAML(raw)
	if err != nil {
		t.Fatalf("decodeTestSuitesFromYAML: %v", err)
	}
	if len(suites) != 2 {
		t.Fatalf("len(suites): got %d want %d", len(suites), 2)
	}
}

func TestDecodeTestSuitesFromYAML_InvalidYAML(t *testing.T) {
	_, err := decodeTestSuitesFromYAML("suite: [\n")
	if err == nil || !strings.Contains(err.Error(), "invalid tests_yaml") {
		t.Fatalf("expected invalid tests_yaml error, got %v", err)
	}
}

func TestDecodeTestSuitesFromYAML_ValidateError(t *testing.T) {
	_, err := decodeTestSuitesFromYAML("suite: x\nprompt: p\ncases: []\n")
	if err == nil {
		t.Fatalf("expected validation error")
	}
}

func TestRespondError_NilErrUsesStatusOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/x", func(c *gin.Context) {
		respondError(c, http.StatusNoContent, nil)
	})

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNoContent)
	}
}

func TestParseLimitParam(t *testing.T) {
	if got, err := parseLimitParam("", 7); err != nil || got != 7 {
		t.Fatalf("parseLimitParam empty: got %d,%v want 7,nil", got, err)
	}
	if _, err := parseLimitParam("wat", 7); err == nil {
		t.Fatalf("expected error for non-int")
	}
	if _, err := parseLimitParam("-1", 7); err == nil {
		t.Fatalf("expected error for <=0")
	}
	if got, err := parseLimitParam("3", 7); err != nil || got != 3 {
		t.Fatalf("parseLimitParam: got %d,%v want 3,nil", got, err)
	}
}

func TestParseTimeParam(t *testing.T) {
	if got, err := parseTimeParam(""); err != nil || !got.IsZero() {
		t.Fatalf("parseTimeParam empty: got %v,%v want zero,nil", got, err)
	}
	if got, err := parseTimeParam("2025-02-03"); err != nil || got.Location() != time.UTC {
		t.Fatalf("parseTimeParam date: got %v,%v", got, err)
	}
	if _, err := parseTimeParam("wat"); err == nil {
		t.Fatalf("expected error for invalid time")
	}
}

func TestFindPromptByName(t *testing.T) {
	if _, err := findPromptByName(nil, ""); err == nil {
		t.Fatalf("expected error for empty name")
	}

	ps := []*prompt.Prompt{{Name: "a"}, {Name: "a"}}
	if _, err := findPromptByName(ps, "a"); err == nil {
		t.Fatalf("expected error for duplicate")
	}

	ps = []*prompt.Prompt{{Name: "a"}}
	if _, err := findPromptByName(ps, "b"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("expected ErrNoRows, got %v", err)
	}
}

func TestFindSuiteByName(t *testing.T) {
	if got := findSuiteByName([]*testcase.TestSuite{{Suite: "a"}}, "a"); got == nil {
		t.Fatalf("expected suite")
	}
	if got := findSuiteByName([]*testcase.TestSuite{{Suite: "a"}}, "b"); got != nil {
		t.Fatalf("expected nil")
	}
}

func TestCompactHelpersAndBuildRunConfig(t *testing.T) {
	prompts := []*prompt.Prompt{nil, {Name: "a"}}
	gotPrompts := compactPrompts(prompts)
	if len(gotPrompts) != 1 {
		t.Fatalf("len(compactPrompts): got %d want %d", len(gotPrompts), 1)
	}

	suites := []*testcase.TestSuite{nil, {Suite: "s"}}
	gotSuites := compactSuites(suites)
	if len(gotSuites) != 1 {
		t.Fatalf("len(compactSuites): got %d want %d", len(gotSuites), 1)
	}

	s := &Server{config: &config.Config{Evaluation: config.EvaluationConfig{Timeout: 123 * time.Millisecond}}}
	cfg := s.buildRunConfig([]string{"p1"}, false, 1, 0.6, 2)
	if cfg["timeout_ms"] != int64(123) {
		t.Fatalf("timeout_ms: got %v want %v", cfg["timeout_ms"], int64(123))
	}
}
