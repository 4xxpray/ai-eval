package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func newTestSQLiteStore(t *testing.T) *SQLiteStore {
	t.Helper()

	path := filepath.Join(t.TempDir(), "store.db")
	st, err := NewSQLiteStore(path)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	t.Cleanup(func() {
		_ = st.Close()
	})
	return st
}

func TestSQLiteStore_SaveRunGetRun(t *testing.T) {
	t.Parallel()

	st := newTestSQLiteStore(t)
	ctx := context.Background()

	start := time.Unix(1_700_000_000, 0).UTC()
	finish := start.Add(2 * time.Minute)

	run := &RunRecord{
		ID:           "run_1",
		StartedAt:    start,
		FinishedAt:   finish,
		TotalSuites:  2,
		PassedSuites: 1,
		FailedSuites: 1,
		Config: map[string]any{
			"trials": 3,
			"output": "json",
		},
	}
	if err := st.SaveRun(ctx, run); err != nil {
		t.Fatalf("SaveRun: %v", err)
	}

	got, err := st.GetRun(ctx, "run_1")
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if got.ID != run.ID {
		t.Fatalf("ID: got %q want %q", got.ID, run.ID)
	}
	if !got.StartedAt.Equal(start) {
		t.Fatalf("StartedAt: got %v want %v", got.StartedAt, start)
	}
	if !got.FinishedAt.Equal(finish) {
		t.Fatalf("FinishedAt: got %v want %v", got.FinishedAt, finish)
	}
	if got.TotalSuites != 2 || got.PassedSuites != 1 || got.FailedSuites != 1 {
		t.Fatalf("Counts: got suites=%d passed=%d failed=%d", got.TotalSuites, got.PassedSuites, got.FailedSuites)
	}
	if got.Config == nil {
		t.Fatalf("Config: expected map")
	}
	if v, ok := got.Config["trials"].(float64); !ok || v != 3 {
		t.Fatalf("Config.trials: got %#v", got.Config["trials"])
	}
	if v, ok := got.Config["output"].(string); !ok || v != "json" {
		t.Fatalf("Config.output: got %#v", got.Config["output"])
	}
}

func TestSQLiteStore_SaveSuiteResultAndGetSuiteResults(t *testing.T) {
	t.Parallel()

	st := newTestSQLiteStore(t)
	ctx := context.Background()

	start := time.Unix(1_700_000_000, 0).UTC()
	run := &RunRecord{
		ID:           "run_2",
		StartedAt:    start,
		FinishedAt:   start.Add(time.Minute),
		TotalSuites:  1,
		PassedSuites: 1,
		FailedSuites: 0,
	}
	if err := st.SaveRun(ctx, run); err != nil {
		t.Fatalf("SaveRun: %v", err)
	}

	suite := &SuiteRecord{
		ID:            "suite_1",
		RunID:         "run_2",
		PromptName:    "p1",
		PromptVersion: "v1",
		SuiteName:     "s1",
		TotalCases:    2,
		PassedCases:   1,
		FailedCases:   1,
		PassRate:      0.5,
		AvgScore:      0.6,
		TotalLatency:  120,
		TotalTokens:   45,
		CreatedAt:     start.Add(2 * time.Minute),
		CaseResults: []CaseRecord{
			{CaseID: "c1", Passed: true, Score: 1, PassAtK: 1, PassExpK: 1, LatencyMs: 50, TokensUsed: 20},
			{CaseID: "c2", Passed: false, Score: 0.2, PassAtK: 0, PassExpK: 0, LatencyMs: 70, TokensUsed: 25, Error: "bad"},
		},
	}
	if err := st.SaveSuiteResult(ctx, suite); err != nil {
		t.Fatalf("SaveSuiteResult: %v", err)
	}

	got, err := st.GetSuiteResults(ctx, "run_2")
	if err != nil {
		t.Fatalf("GetSuiteResults: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len: got %d want %d", len(got), 1)
	}
	if got[0].SuiteName != "s1" || got[0].PromptName != "p1" || got[0].PromptVersion != "v1" {
		t.Fatalf("Suite: got %#v", got[0])
	}
	if len(got[0].CaseResults) != 2 {
		t.Fatalf("CaseResults: got %d want %d", len(got[0].CaseResults), 2)
	}
	if got[0].CaseResults[1].Error != "bad" {
		t.Fatalf("CaseResults[1].Error: got %q want %q", got[0].CaseResults[1].Error, "bad")
	}
}

func TestSQLiteStore_ListRuns_Filter(t *testing.T) {
	t.Parallel()

	st := newTestSQLiteStore(t)
	ctx := context.Background()

	t0 := time.Unix(1_700_000_000, 0).UTC()
	run1 := &RunRecord{
		ID:           "run_a",
		StartedAt:    t0,
		FinishedAt:   t0.Add(time.Minute),
		TotalSuites:  1,
		PassedSuites: 1,
		FailedSuites: 0,
	}
	run2 := &RunRecord{
		ID:           "run_b",
		StartedAt:    t0.Add(2 * time.Hour),
		FinishedAt:   t0.Add(2*time.Hour + time.Minute),
		TotalSuites:  1,
		PassedSuites: 0,
		FailedSuites: 1,
	}

	if err := st.SaveRun(ctx, run1); err != nil {
		t.Fatalf("SaveRun run1: %v", err)
	}
	if err := st.SaveRun(ctx, run2); err != nil {
		t.Fatalf("SaveRun run2: %v", err)
	}

	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_a",
		RunID:         "run_a",
		PromptName:    "p1",
		PromptVersion: "v1",
		SuiteName:     "s1",
		TotalCases:    1,
		PassedCases:   1,
		FailedCases:   0,
		PassRate:      1,
		AvgScore:      1,
		TotalLatency:  10,
		TotalTokens:   5,
		CreatedAt:     t0.Add(10 * time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult run1: %v", err)
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_b",
		RunID:         "run_b",
		PromptName:    "p2",
		PromptVersion: "v2",
		SuiteName:     "s2",
		TotalCases:    1,
		PassedCases:   0,
		FailedCases:   1,
		PassRate:      0,
		AvgScore:      0,
		TotalLatency:  10,
		TotalTokens:   5,
		CreatedAt:     t0.Add(2*time.Hour + 10*time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult run2: %v", err)
	}

	runs, err := st.ListRuns(ctx, RunFilter{PromptName: "p1", Limit: 10})
	if err != nil {
		t.Fatalf("ListRuns: %v", err)
	}
	if len(runs) != 1 || runs[0].ID != "run_a" {
		t.Fatalf("ListRuns prompt filter: got %#v", runs)
	}

	runs, err = st.ListRuns(ctx, RunFilter{Since: t0.Add(time.Hour), Limit: 10})
	if err != nil {
		t.Fatalf("ListRuns since: %v", err)
	}
	if len(runs) != 1 || runs[0].ID != "run_b" {
		t.Fatalf("ListRuns since: got %#v", runs)
	}
}

func TestSQLiteStore_GetPromptHistory(t *testing.T) {
	t.Parallel()

	st := newTestSQLiteStore(t)
	ctx := context.Background()
	t0 := time.Unix(1_700_000_000, 0).UTC()

	if err := st.SaveRun(ctx, &RunRecord{
		ID:           "run_h1",
		StartedAt:    t0,
		FinishedAt:   t0.Add(time.Minute),
		TotalSuites:  1,
		PassedSuites: 1,
		FailedSuites: 0,
	}); err != nil {
		t.Fatalf("SaveRun: %v", err)
	}
	if err := st.SaveRun(ctx, &RunRecord{
		ID:           "run_h2",
		StartedAt:    t0.Add(2 * time.Hour),
		FinishedAt:   t0.Add(2*time.Hour + time.Minute),
		TotalSuites:  1,
		PassedSuites: 1,
		FailedSuites: 0,
	}); err != nil {
		t.Fatalf("SaveRun: %v", err)
	}

	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_h1",
		RunID:         "run_h1",
		PromptName:    "p1",
		PromptVersion: "v1",
		SuiteName:     "s1",
		TotalCases:    1,
		PassedCases:   1,
		FailedCases:   0,
		PassRate:      1,
		AvgScore:      1,
		TotalLatency:  10,
		TotalTokens:   5,
		CreatedAt:     t0.Add(10 * time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult: %v", err)
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_h2",
		RunID:         "run_h2",
		PromptName:    "p1",
		PromptVersion: "v1",
		SuiteName:     "s2",
		TotalCases:    1,
		PassedCases:   1,
		FailedCases:   0,
		PassRate:      1,
		AvgScore:      1,
		TotalLatency:  10,
		TotalTokens:   5,
		CreatedAt:     t0.Add(2*time.Hour + 10*time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult: %v", err)
	}

	history, err := st.GetPromptHistory(ctx, "p1", 1)
	if err != nil {
		t.Fatalf("GetPromptHistory: %v", err)
	}
	if len(history) != 1 || history[0].ID != "suite_h2" {
		t.Fatalf("GetPromptHistory: got %#v", history)
	}
}

func TestSQLiteStore_GetVersionComparison(t *testing.T) {
	t.Parallel()

	st := newTestSQLiteStore(t)
	ctx := context.Background()
	t0 := time.Unix(1_700_000_000, 0).UTC()

	if err := st.SaveRun(ctx, &RunRecord{
		ID:           "run_v1",
		StartedAt:    t0,
		FinishedAt:   t0.Add(time.Minute),
		TotalSuites:  1,
		PassedSuites: 1,
		FailedSuites: 0,
	}); err != nil {
		t.Fatalf("SaveRun v1: %v", err)
	}
	if err := st.SaveRun(ctx, &RunRecord{
		ID:           "run_v2",
		StartedAt:    t0.Add(2 * time.Hour),
		FinishedAt:   t0.Add(2*time.Hour + time.Minute),
		TotalSuites:  1,
		PassedSuites: 1,
		FailedSuites: 0,
	}); err != nil {
		t.Fatalf("SaveRun v2: %v", err)
	}

	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_v1",
		RunID:         "run_v1",
		PromptName:    "p1",
		PromptVersion: "v1",
		SuiteName:     "s1",
		TotalCases:    2,
		PassedCases:   1,
		FailedCases:   1,
		PassRate:      0.5,
		AvgScore:      0.6,
		TotalLatency:  10,
		TotalTokens:   5,
		CreatedAt:     t0.Add(10 * time.Second),
		CaseResults: []CaseRecord{
			{CaseID: "c1", Passed: true},
			{CaseID: "c2", Passed: false},
		},
	}); err != nil {
		t.Fatalf("SaveSuiteResult v1: %v", err)
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_v2",
		RunID:         "run_v2",
		PromptName:    "p1",
		PromptVersion: "v2",
		SuiteName:     "s1",
		TotalCases:    2,
		PassedCases:   1,
		FailedCases:   1,
		PassRate:      0.5,
		AvgScore:      0.6,
		TotalLatency:  10,
		TotalTokens:   5,
		CreatedAt:     t0.Add(2*time.Hour + 10*time.Second),
		CaseResults: []CaseRecord{
			{CaseID: "c1", Passed: false},
			{CaseID: "c2", Passed: true},
		},
	}); err != nil {
		t.Fatalf("SaveSuiteResult v2: %v", err)
	}

	comp, err := st.GetVersionComparison(ctx, "p1", "v1", "v2")
	if err != nil {
		t.Fatalf("GetVersionComparison: %v", err)
	}
	if len(comp.V1Results) != 1 || len(comp.V2Results) != 1 {
		t.Fatalf("Results: got v1=%d v2=%d", len(comp.V1Results), len(comp.V2Results))
	}
	if len(comp.Regressions) != 1 || comp.Regressions[0] != "c1" {
		t.Fatalf("Regressions: got %#v", comp.Regressions)
	}
	if len(comp.Improvements) != 1 || comp.Improvements[0] != "c2" {
		t.Fatalf("Improvements: got %#v", comp.Improvements)
	}
}
