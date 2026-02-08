package api

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/store"
)

func TestHandlers_ListRuns_NotInitialized(t *testing.T) {
	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_ListRuns_InvalidUntil(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs?until=wat", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_ListRuns_StoreError(t *testing.T) {
	st := &fakeStore{
		ListRunsFunc: func(ctx context.Context, filter store.RunFilter) ([]*store.RunRecord, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetRun_NotInitialized(t *testing.T) {
	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetRun_MissingID(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/%20", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetRun_StoreError(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetRunResults_MissingID(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/%20/results", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetRunResults_NotInitialized(t *testing.T) {
	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1/results", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetRunResults_NotFound(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return nil, sql.ErrNoRows
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1/results", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlers_GetRunResults_GetRunError(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1/results", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetRunResults_GetSuiteResultsError(t *testing.T) {
	st := &fakeStore{
		GetRunFunc: func(ctx context.Context, id string) (*store.RunRecord, error) {
			return &store.RunRecord{ID: id}, nil
		},
		GetSuiteResultsFunc: func(ctx context.Context, runID string) ([]*store.SuiteRecord, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/runs/r1/results", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetPromptHistory_MissingPromptName(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/history/%20", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetPromptHistory_NotInitialized(t *testing.T) {
	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/history/p", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_GetPromptHistory_InvalidLimit(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/history/p?limit=wat", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_GetPromptHistory_StoreError(t *testing.T) {
	st := &fakeStore{
		GetPromptHistoryFunc: func(ctx context.Context, promptName string, limit int) ([]*store.SuiteRecord, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodGet, "/api/history/p", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_CompareVersions_BadJSON(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/compare", bytes.NewBufferString("{"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_CompareVersions_MissingFields(t *testing.T) {
	s := &Server{store: &fakeStore{}}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/compare", bytes.NewBufferString(`{"prompt":"p","v1":"a"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestHandlers_CompareVersions_StoreError(t *testing.T) {
	st := &fakeStore{
		GetVersionComparisonFunc: func(ctx context.Context, promptName, v1, v2 string) (*store.VersionComparison, error) {
			return nil, errors.New("boom")
		},
	}
	s := &Server{store: st}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/compare", bytes.NewBufferString(`{"prompt":"p","v1":"a","v2":"b"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestHandlers_CompareVersions_NotInitialized(t *testing.T) {
	s := &Server{}
	r := newTestRouterForServer(t, s)

	req := httptest.NewRequest(http.MethodPost, "/api/compare", bytes.NewBufferString(`{"prompt":"p","v1":"a","v2":"b"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status: got %d want %d", rec.Code, http.StatusInternalServerError)
	}
}
