package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func setupAPITestWorkspace(t *testing.T) {
	t.Helper()

	dir := t.TempDir()
	promptsPath := filepath.Join(dir, promptsDir)
	testsPath := filepath.Join(dir, testsDir)

	if err := os.MkdirAll(promptsPath, 0o755); err != nil {
		t.Fatalf("MkdirAll prompts: %v", err)
	}
	if err := os.MkdirAll(testsPath, 0o755); err != nil {
		t.Fatalf("MkdirAll tests: %v", err)
	}

	promptPayload := []byte("name: example\nversion: v1\ndescription: example prompt\ntemplate: hello\n")
	if err := os.WriteFile(filepath.Join(promptsPath, "example.yaml"), promptPayload, 0o644); err != nil {
		t.Fatalf("WriteFile prompt: %v", err)
	}

	suitePayload := []byte("suite: basic\nprompt: example\ncases:\n  - id: c1\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n")
	if err := os.WriteFile(filepath.Join(testsPath, "basic.yaml"), suitePayload, 0o644); err != nil {
		t.Fatalf("WriteFile suite: %v", err)
	}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(cwd)
	})
}

func newTestRouter(t *testing.T) *gin.Engine {
	t.Helper()

	gin.SetMode(gin.TestMode)
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	r := gin.New()
	s := &Server{router: r}
	if err := s.registerRoutes(); err != nil {
		t.Fatalf("registerRoutes: %v", err)
	}
	return r
}

func TestHandlers_Health(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var body map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if body["status"] != "ok" {
		t.Fatalf("status field: got %v want %q", body["status"], "ok")
	}
}

func TestHandlers_ListPrompts(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/prompts", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out []prompt.Prompt
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len(prompts): got %d want %d", len(out), 1)
	}
	if out[0].Name != "example" {
		t.Fatalf("prompt[0].Name: got %q want %q", out[0].Name, "example")
	}
}

func TestHandlers_ListTests(t *testing.T) {
	setupAPITestWorkspace(t)
	r := newTestRouter(t)

	req := httptest.NewRequest(http.MethodGet, "/api/tests", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out []testcase.TestSuite
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len(tests): got %d want %d", len(out), 1)
	}
	if out[0].Suite != "basic" {
		t.Fatalf("suite[0].Suite: got %q want %q", out[0].Suite, "basic")
	}
}

func TestPromptFileName(t *testing.T) {
	tests := []struct {
		name    string
		want    string
		wantErr bool
	}{
		{name: "example", want: "example.yaml"},
		{name: "  spaced  ", want: "spaced.yaml"},
		{name: "foo.bar", want: "foo.bar.yaml"},
		{name: "", wantErr: true},
		{name: "   ", wantErr: true},
		{name: ".hidden", wantErr: true},
		{name: ".", wantErr: true},
		{name: "a/b", wantErr: true},
		{name: "a\\b", wantErr: true},
		{name: "a:b", wantErr: true},
		{name: "a*b", wantErr: true},
		{name: "a?b", wantErr: true},
		{name: "a\"b", wantErr: true},
		{name: "a<b", wantErr: true},
		{name: "a>b", wantErr: true},
		{name: "a|b", wantErr: true},
	}

	for _, tc := range tests {
		got, err := promptFileName(tc.name)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("promptFileName(%q): expected error", tc.name)
			}
			continue
		}
		if err != nil {
			t.Fatalf("promptFileName(%q): %v", tc.name, err)
		}
		if got != tc.want {
			t.Fatalf("promptFileName(%q): got %q want %q", tc.name, got, tc.want)
		}
	}
}
