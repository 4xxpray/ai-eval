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

func TestStaticHandler_ApiPathsReturnNotFoundJSON(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodGet, "/api", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
	if !strings.Contains(rec.Body.String(), "not found") {
		t.Fatalf("expected not found JSON")
	}
}

func TestStaticHandler_NonExistentRouteServesIndex(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodGet, "/some/route", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	if !strings.Contains(rec.Body.String(), "<title>AI Eval Dashboard</title>") {
		t.Fatalf("expected index content")
	}
}

func TestStaticHandler_ServesExistingFile(t *testing.T) {
	ensureStaticRoot(t)

	path := filepath.Join(staticRoot, "codex_static_test.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(path) })

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodGet, "/codex_static_test.txt", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	if strings.TrimSpace(rec.Body.String()) != "hello" {
		t.Fatalf("body: got %q want %q", rec.Body.String(), "hello")
	}
}

func TestStaticHandler_PathDotServesIndex(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodGet, "/.", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	if !strings.Contains(rec.Body.String(), "<title>AI Eval Dashboard</title>") {
		t.Fatalf("expected index content")
	}
}

func TestStaticHandler_Head(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodHead, "/", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestStaticHandler_NonGetHeadReturnsNotFound(t *testing.T) {
	ensureStaticRoot(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerStatic()

	req := httptest.NewRequest(http.MethodPost, "/x", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestRegisterStatic_NilSafe(t *testing.T) {
	var nilServer *Server
	nilServer.registerStatic()

	s := &Server{}
	s.registerStatic()
}
