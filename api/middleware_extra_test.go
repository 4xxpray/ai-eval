package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRegisterMiddleware_NilSafe(t *testing.T) {
	var nilServer *Server
	nilServer.registerMiddleware()

	s := &Server{}
	s.registerMiddleware()
}

func TestRegisterMiddleware_AttachesRecoveryAndCORS(t *testing.T) {
	t.Setenv("AI_EVAL_CORS_ORIGINS", "*")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{router: r}
	s.registerMiddleware()

	r.GET("/panic", func(c *gin.Context) { panic("boom") })

	req := httptest.NewRequest(http.MethodGet, "/panic", nil)
	req.Header.Set("Origin", "http://example.com")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want %q", got, "*")
	}
}

func TestCorsMiddleware_IgnoresEmptyOriginsList(t *testing.T) {
	t.Setenv("AI_EVAL_CORS_ORIGINS", ", ,")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(corsMiddleware())
	r.GET("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("Origin", "http://example.com")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want empty", got)
	}
}

func TestCorsMiddleware_OptionsAlwaysNoContentWithOrigin(t *testing.T) {
	t.Setenv("AI_EVAL_CORS_ORIGINS", "http://allowed.example")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(corsMiddleware())
	r.GET("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodOptions, "/x", nil)
	req.Header.Set("Origin", "http://evil.example")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNoContent)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin: got %q want empty", got)
	}
}

func TestAPIKeyAuthMiddleware_EmptyExpectedAllowsAll(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(apiKeyAuthMiddleware(""))
	r.GET("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}

func TestAPIKeyAuthMiddleware_OptionsBypass(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(apiKeyAuthMiddleware("secret"))
	r.OPTIONS("/x", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	req := httptest.NewRequest(http.MethodOptions, "/x", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}
}
