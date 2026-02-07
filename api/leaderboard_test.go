package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
)

func newTestRouterWithLeaderboard(t *testing.T, lb *leaderboard.Store) *gin.Engine {
	t.Helper()

	gin.SetMode(gin.TestMode)
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	r := gin.New()
	s := &Server{router: r, lbStore: lb}
	if err := s.registerRoutes(); err != nil {
		t.Fatalf("registerRoutes: %v", err)
	}
	return r
}

func TestLeaderboardHandlers_GetLeaderboard(t *testing.T) {
	lb, err := leaderboard.NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer lb.Close()

	ctx := context.Background()
	if err := lb.Save(ctx, &leaderboard.Entry{
		Model:    "m1",
		Provider: "openai",
		Dataset:  "mmlu",
		Score:    0.80,
		Accuracy: 0.80,
		Latency:  100,
		Cost:     0,
		EvalDate: time.UnixMilli(1000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	r := newTestRouterWithLeaderboard(t, lb)
	req := httptest.NewRequest(http.MethodGet, "/api/leaderboard?dataset=mmlu&limit=10", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out []leaderboard.Entry
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len(entries): got %d want %d", len(out), 1)
	}
	if out[0].Model != "m1" {
		t.Fatalf("entry[0].Model: got %q want %q", out[0].Model, "m1")
	}
}

func TestLeaderboardHandlers_GetLeaderboard_MissingDataset(t *testing.T) {
	lb, err := leaderboard.NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer lb.Close()

	r := newTestRouterWithLeaderboard(t, lb)
	req := httptest.NewRequest(http.MethodGet, "/api/leaderboard", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestLeaderboardHandlers_GetModelHistory(t *testing.T) {
	lb, err := leaderboard.NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer lb.Close()

	ctx := context.Background()
	if err := lb.Save(ctx, &leaderboard.Entry{
		Model:    "m1",
		Provider: "claude",
		Dataset:  "gsm8k",
		Score:    0.70,
		Accuracy: 0.70,
		Latency:  50,
		Cost:     0,
		EvalDate: time.UnixMilli(1000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := lb.Save(ctx, &leaderboard.Entry{
		Model:    "m1",
		Provider: "claude",
		Dataset:  "gsm8k",
		Score:    0.90,
		Accuracy: 0.90,
		Latency:  70,
		Cost:     0,
		EvalDate: time.UnixMilli(2000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	r := newTestRouterWithLeaderboard(t, lb)
	req := httptest.NewRequest(http.MethodGet, "/api/leaderboard/history?model=m1&dataset=gsm8k", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusOK)
	}

	var out []leaderboard.Entry
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("len(entries): got %d want %d", len(out), 2)
	}
	if out[0].Score != 0.90 {
		t.Fatalf("history[0].Score: got %.2f want %.2f", out[0].Score, 0.90)
	}
}
