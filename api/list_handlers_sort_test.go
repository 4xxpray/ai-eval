package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestHandlers_ListPrompts_SortsWhenMultiple(t *testing.T) {
	setupAPITestWorkspace(t)

	payload := []byte("name: zzz\nversion: v1\ndescription: z\ntemplate: hi\n")
	if err := os.WriteFile(filepath.Join(promptsDir, "zzz.yaml"), payload, 0o644); err != nil {
		t.Fatalf("WriteFile prompt: %v", err)
	}

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
	if len(out) != 2 {
		t.Fatalf("len(prompts): got %d want %d", len(out), 2)
	}
	if out[0].Name != "example" || out[1].Name != "zzz" {
		t.Fatalf("order: got %q,%q want %q,%q", out[0].Name, out[1].Name, "example", "zzz")
	}
}

func TestHandlers_ListTests_SortsWhenMultiple(t *testing.T) {
	setupAPITestWorkspace(t)

	suitePayload := []byte("suite: zzz\nprompt: example\ncases:\n  - id: c2\n    input:\n      text: hi\n    expected:\n      contains:\n        - ok\n")
	if err := os.WriteFile(filepath.Join(testsDir, "zzz.yaml"), suitePayload, 0o644); err != nil {
		t.Fatalf("WriteFile suite: %v", err)
	}

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
	if len(out) != 2 {
		t.Fatalf("len(suites): got %d want %d", len(out), 2)
	}
	if out[0].Suite != "basic" || out[1].Suite != "zzz" {
		t.Fatalf("order: got %q,%q want %q,%q", out[0].Suite, out[1].Suite, "basic", "zzz")
	}
}
