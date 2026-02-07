package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRegisterRoutes_RequiresExplicitAuthConfig(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	if err := s.registerRoutes(); err == nil {
		t.Fatalf("registerRoutes: expected error")
	}
}

func TestRegisterRoutes_AllowsDisableAuthOptOut(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	if err := s.registerRoutes(); err != nil {
		t.Fatalf("registerRoutes: %v", err)
	}
}

func TestRegisterRoutes_APIKeyEnforcesAuth(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "secret")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	if err := s.registerRoutes(); err != nil {
		t.Fatalf("registerRoutes: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("GET /api/health without key: got %d want %d", rec.Code, http.StatusUnauthorized)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/health", nil)
	req.Header.Set("X-API-Key", "wrong")
	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("GET /api/health wrong key: got %d want %d", rec.Code, http.StatusUnauthorized)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/health", nil)
	req.Header.Set("X-API-Key", "secret")
	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/health correct key: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestCorsMiddleware_DefaultNoCORS(t *testing.T) {
	t.Setenv("AI_EVAL_CORS_ORIGINS", "")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(corsMiddleware())
	r.GET("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("Origin", "http://evil.example")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want empty", got)
	}
}

func TestCorsMiddleware_AllowsConfiguredOrigin(t *testing.T) {
	t.Setenv("AI_EVAL_CORS_ORIGINS", "http://example.com, http://localhost:3000")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(corsMiddleware())
	r.GET("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("Origin", "http://example.com")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://example.com" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want %q", got, "http://example.com")
	}
	if got := rec.Header().Get("Vary"); got != "Origin" {
		t.Fatalf("Vary: got %q want %q", got, "Origin")
	}

	req = httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("Origin", "http://evil.example")
	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want empty", got)
	}

	req = httptest.NewRequest(http.MethodOptions, "/x", nil)
	req.Header.Set("Origin", "http://example.com")
	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("OPTIONS /x: got %d want %d", rec.Code, http.StatusNoContent)
	}
}

func TestCorsMiddleware_WildcardAllowsAll(t *testing.T) {
	t.Setenv("AI_EVAL_CORS_ORIGINS", "*")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(corsMiddleware())
	r.GET("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("Origin", "http://evil.example")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want %q", got, "*")
	}
}
