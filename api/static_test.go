package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func ensureStaticRoot(t *testing.T) {
	t.Helper()

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if _, err := os.Stat(filepath.Join(cwd, staticRoot, "index.html")); err == nil {
		return
	}

	root := filepath.Dir(cwd)
	if _, err := os.Stat(filepath.Join(root, staticRoot, "index.html")); err != nil {
		t.Fatalf("static root: %v", err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(cwd)
	})
}

func TestStaticHandler_ServesIndexFile(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	if !strings.Contains(rec.Body.String(), "<title>AI Eval Dashboard</title>") {
		t.Fatalf("body: expected index content")
	}
}

func TestStaticHandler_RejectsTraversal(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	paths := []string{
		"/../api/handlers.go",
		"/..%2fapi/handlers.go",
		"/%2e%2e/api/handlers.go",
		"/..\\api\\handlers.go",
	}

	for _, path := range paths {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden && rec.Code != http.StatusNotFound && rec.Code != http.StatusBadRequest {
			t.Fatalf("path %q: got %d want 400, 403, or 404", path, rec.Code)
		}
	}
}
