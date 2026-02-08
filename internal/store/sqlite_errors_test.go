package store

import (
	"context"
	"database/sql"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNewSQLiteStore_Errors(t *testing.T) {
	if _, err := NewSQLiteStore("   "); err == nil {
		t.Fatalf("NewSQLiteStore(empty): expected error")
	}

	dir := t.TempDir()
	notADir := filepath.Join(dir, "notadir")
	if err := os.WriteFile(notADir, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := NewSQLiteStore(filepath.Join(notADir, "db.sqlite")); err == nil {
		t.Fatalf("NewSQLiteStore(mkdir): expected error")
	}
}

func TestNewSQLiteStore_PingError(t *testing.T) {
	dir := t.TempDir()
	if _, err := NewSQLiteStore(dir); err == nil {
		t.Fatalf("NewSQLiteStore(directory): expected error")
	}
}

func TestNewSQLiteStore_InitSchemaError_ReadOnlyDSN(t *testing.T) {
	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	tmp := t.TempDir()
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWd) })

	db, err := sql.Open("sqlite3", "ro.db")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		t.Fatalf("Ping: %v", err)
	}
	_ = db.Close()

	if _, err := NewSQLiteStore("file:ro.db?mode=ro"); err == nil {
		t.Fatalf("NewSQLiteStore(read-only): expected error")
	}
}

func TestInitSQLiteSchema_ClosedDB(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := initSQLiteSchema(db); err == nil {
		t.Fatalf("initSQLiteSchema: expected error for closed db")
	}
}

func TestSQLiteStore_NilReceiver(t *testing.T) {
	if err := (*SQLiteStore)(nil).Close(); err != nil {
		t.Fatalf("Close(nil): %v", err)
	}
	if err := (&SQLiteStore{}).Close(); err != nil {
		t.Fatalf("Close(nil db): %v", err)
	}
	if err := (*SQLiteStore)(nil).prepareStatements(); err == nil {
		t.Fatalf("prepareStatements(nil): expected error")
	}

	if err := (*SQLiteStore)(nil).SaveRun(context.Background(), &RunRecord{ID: "x"}); err == nil {
		t.Fatalf("SaveRun(nil store): expected error")
	}
	if err := (*SQLiteStore)(nil).SaveSuiteResult(context.Background(), &SuiteRecord{ID: "x"}); err == nil {
		t.Fatalf("SaveSuiteResult(nil store): expected error")
	}
	if _, err := (*SQLiteStore)(nil).GetRun(context.Background(), "x"); err == nil {
		t.Fatalf("GetRun(nil store): expected error")
	}
	if _, err := (*SQLiteStore)(nil).ListRuns(context.Background(), RunFilter{}); err == nil {
		t.Fatalf("ListRuns(nil store): expected error")
	}
	if _, err := (*SQLiteStore)(nil).GetSuiteResults(context.Background(), "x"); err == nil {
		t.Fatalf("GetSuiteResults(nil store): expected error")
	}
	if _, err := (*SQLiteStore)(nil).GetPromptHistory(context.Background(), "p", 1); err == nil {
		t.Fatalf("GetPromptHistory(nil store): expected error")
	}
	if _, err := (*SQLiteStore)(nil).GetVersionComparison(context.Background(), "p", "v1", "v2"); err == nil {
		t.Fatalf("GetVersionComparison(nil store): expected error")
	}
}

func TestSQLiteStore_SaveRun_ValidationAndDBErrors(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if err := st.SaveRun(nil, &RunRecord{ID: "x"}); err == nil {
		t.Fatalf("SaveRun(nil ctx): expected error")
	}
	if err := st.SaveRun(ctx, nil); err == nil {
		t.Fatalf("SaveRun(nil run): expected error")
	}

	t0 := time.Unix(1_700_000_000, 0).UTC()
	if err := st.SaveRun(ctx, &RunRecord{ID: "  ", StartedAt: t0, FinishedAt: t0.Add(time.Minute)}); err == nil {
		t.Fatalf("SaveRun(empty id): expected error")
	}
	if err := st.SaveRun(ctx, &RunRecord{ID: "run", StartedAt: time.Time{}, FinishedAt: time.Time{}}); err == nil {
		t.Fatalf("SaveRun(missing timestamps): expected error")
	}

	if err := st.SaveRun(ctx, &RunRecord{
		ID:         "run_badcfg",
		StartedAt:  t0,
		FinishedAt: t0.Add(time.Minute),
		Config: map[string]any{
			"bad": make(chan int),
		},
	}); err == nil {
		t.Fatalf("SaveRun(marshal config): expected error")
	}

	if _, err := st.db.ExecContext(ctx, `DROP TABLE runs`); err != nil {
		t.Fatalf("DROP TABLE runs: %v", err)
	}
	if err := st.SaveRun(ctx, &RunRecord{
		ID:         "run_missing_table",
		StartedAt:  t0,
		FinishedAt: t0.Add(time.Minute),
	}); err == nil {
		t.Fatalf("SaveRun(insert error): expected error")
	}

	st2 := newTestSQLiteStore(t)
	if err := st2.db.Close(); err != nil {
		t.Fatalf("Close db: %v", err)
	}
	if err := st2.SaveRun(ctx, &RunRecord{ID: "x", StartedAt: t0, FinishedAt: t0.Add(time.Minute)}); err == nil {
		t.Fatalf("SaveRun(closed db): expected error")
	}
}

func TestSQLiteStore_SaveSuiteResult_ValidationAndErrors(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if err := st.SaveSuiteResult(nil, &SuiteRecord{ID: "x"}); err == nil {
		t.Fatalf("SaveSuiteResult(nil ctx): expected error")
	}
	if err := st.SaveSuiteResult(ctx, nil); err == nil {
		t.Fatalf("SaveSuiteResult(nil result): expected error")
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{ID: "  "}); err == nil {
		t.Fatalf("SaveSuiteResult(empty id): expected error")
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{ID: "s", RunID: " "}); err == nil {
		t.Fatalf("SaveSuiteResult(empty run id): expected error")
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{ID: "s", RunID: "r", PromptName: " ", SuiteName: "s"}); err == nil {
		t.Fatalf("SaveSuiteResult(missing prompt/suite): expected error")
	}

	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:         "suite_badcase",
		RunID:      "run",
		PromptName: "p",
		SuiteName:  "s",
		CaseResults: []CaseRecord{
			{CaseID: "c1", Score: math.NaN()},
		},
	}); err == nil {
		t.Fatalf("SaveSuiteResult(marshal NaN): expected error")
	}

	if _, err := st.db.ExecContext(ctx, `DROP TABLE suite_results`); err != nil {
		t.Fatalf("DROP TABLE suite_results: %v", err)
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:         "suite_missing_table",
		RunID:      "run",
		PromptName: "p",
		SuiteName:  "s",
		CreatedAt:  time.Now(),
	}); err == nil {
		t.Fatalf("SaveSuiteResult(insert error): expected error")
	}

	st2 := newTestSQLiteStore(t)
	if err := st2.db.Close(); err != nil {
		t.Fatalf("Close db: %v", err)
	}
	if err := st2.SaveSuiteResult(ctx, &SuiteRecord{
		ID:         "suite_closed",
		RunID:      "run",
		PromptName: "p",
		SuiteName:  "s",
	}); err == nil {
		t.Fatalf("SaveSuiteResult(begin tx): expected error")
	}
}

func TestSQLiteStore_GetRun_Errors(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if _, err := st.GetRun(nil, "x"); err == nil {
		t.Fatalf("GetRun(nil ctx): expected error")
	}
	if _, err := st.GetRun(ctx, " "); err == nil {
		t.Fatalf("GetRun(empty id): expected error")
	}
	if _, err := st.GetRun(ctx, "missing"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("GetRun(missing): got %v want sql.ErrNoRows", err)
	}

	if _, err := st.db.ExecContext(ctx, `
		INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json)
		VALUES ('badcfg', 1, 2, 0, 0, 0, '{bad')
	`); err != nil {
		t.Fatalf("INSERT bad cfg: %v", err)
	}
	if _, err := st.GetRun(ctx, "badcfg"); err == nil {
		t.Fatalf("GetRun(invalid config): expected error")
	}

	st2 := newTestSQLiteStore(t)
	if err := st2.db.Close(); err != nil {
		t.Fatalf("Close db: %v", err)
	}
	if _, err := st2.GetRun(ctx, "x"); err == nil || errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("GetRun(scan error): %v", err)
	}
}

func TestSQLiteStore_ListRuns_ErrorsAndFilters(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()
	t0 := time.Unix(1_700_000_000, 0).UTC()

	if _, err := st.ListRuns(nil, RunFilter{}); err == nil {
		t.Fatalf("ListRuns(nil ctx): expected error")
	}

	run1 := &RunRecord{ID: "run1", StartedAt: t0, FinishedAt: t0.Add(time.Minute)}
	run2 := &RunRecord{ID: "run2", StartedAt: t0.Add(2 * time.Hour), FinishedAt: t0.Add(2*time.Hour + time.Minute)}
	if err := st.SaveRun(ctx, run1); err != nil {
		t.Fatalf("SaveRun run1: %v", err)
	}
	if err := st.SaveRun(ctx, run2); err != nil {
		t.Fatalf("SaveRun run2: %v", err)
	}

	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite1",
		RunID:         "run1",
		PromptName:    "p",
		PromptVersion: "v1",
		SuiteName:     "s",
		CreatedAt:     t0.Add(10 * time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult suite1: %v", err)
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite2",
		RunID:         "run2",
		PromptName:    "p",
		PromptVersion: "v2",
		SuiteName:     "s",
		CreatedAt:     t0.Add(2*time.Hour + 10*time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult suite2: %v", err)
	}

	runs, err := st.ListRuns(ctx, RunFilter{PromptVersion: "v2", Limit: 0})
	if err != nil {
		t.Fatalf("ListRuns(version filter): %v", err)
	}
	if len(runs) != 1 || runs[0].ID != "run2" {
		t.Fatalf("ListRuns(version filter): got %#v", runs)
	}

	runs, err = st.ListRuns(ctx, RunFilter{Until: t0.Add(time.Hour), Limit: 50})
	if err != nil {
		t.Fatalf("ListRuns(until): %v", err)
	}
	if len(runs) != 1 || runs[0].ID != "run1" {
		t.Fatalf("ListRuns(until): got %#v", runs)
	}

	st2 := newTestSQLiteStore(t)
	if err := st2.db.Close(); err != nil {
		t.Fatalf("Close db: %v", err)
	}
	if _, err := st2.ListRuns(ctx, RunFilter{}); err == nil {
		t.Fatalf("ListRuns(closed db): expected error")
	}
}

func TestSQLiteStore_ListRuns_ScanAndDecodeErrors(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if _, err := st.db.ExecContext(ctx, `
		INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json)
		VALUES ('badscan', 'x', 1, 0, 0, 0, NULL)
	`); err != nil {
		t.Fatalf("INSERT badscan: %v", err)
	}
	if _, err := st.ListRuns(ctx, RunFilter{Limit: 10}); err == nil || !strings.Contains(err.Error(), "scan run") {
		t.Fatalf("ListRuns(scan): %v", err)
	}

	st2 := newTestSQLiteStore(t)
	if _, err := st2.db.ExecContext(ctx, `
		INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json)
		VALUES ('badcfg', 1, 2, 0, 0, 0, '{bad')
	`); err != nil {
		t.Fatalf("INSERT badcfg: %v", err)
	}
	if _, err := st2.ListRuns(ctx, RunFilter{Limit: 10}); err == nil || !strings.Contains(err.Error(), "decode run config") {
		t.Fatalf("ListRuns(decode): %v", err)
	}
}

func TestSQLiteStore_QueryValidationErrors(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if _, err := st.GetSuiteResults(nil, "x"); err == nil {
		t.Fatalf("GetSuiteResults(nil ctx): expected error")
	}
	if _, err := st.GetSuiteResults(ctx, " "); err == nil {
		t.Fatalf("GetSuiteResults(empty run id): expected error")
	}

	if _, err := st.GetPromptHistory(nil, "p", 1); err == nil {
		t.Fatalf("GetPromptHistory(nil ctx): expected error")
	}
	if _, err := st.GetPromptHistory(ctx, "  ", 1); err == nil {
		t.Fatalf("GetPromptHistory(empty prompt): expected error")
	}
	if _, err := st.GetPromptHistory(ctx, "p", 0); err != nil {
		t.Fatalf("GetPromptHistory(default limit): %v", err)
	}

	if _, err := st.GetVersionComparison(nil, "p", "v1", "v2"); err == nil {
		t.Fatalf("GetVersionComparison(nil ctx): expected error")
	}
	if _, err := st.GetVersionComparison(ctx, "p", " ", "v2"); err == nil {
		t.Fatalf("GetVersionComparison(missing): expected error")
	}
	if _, err := st.GetVersionComparison(ctx, "p", "v1", "v2"); err == nil {
		t.Fatalf("GetVersionComparison(no runs): expected error")
	}

	if err := st.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := st.GetSuiteResults(ctx, "run"); err == nil {
		t.Fatalf("GetSuiteResults(closed stmt): expected error")
	}
	if _, err := st.GetPromptHistory(ctx, "p", 1); err == nil {
		t.Fatalf("GetPromptHistory(closed stmt): expected error")
	}
}

func TestSQLiteStore_GetVersionComparison_SecondLatestRunFails(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()
	t0 := time.Unix(1_700_000_000, 0).UTC()

	if err := st.SaveRun(ctx, &RunRecord{ID: "run_v1", StartedAt: t0, FinishedAt: t0.Add(time.Minute)}); err != nil {
		t.Fatalf("SaveRun: %v", err)
	}
	if err := st.SaveSuiteResult(ctx, &SuiteRecord{
		ID:            "suite_v1",
		RunID:         "run_v1",
		PromptName:    "p",
		PromptVersion: "v1",
		SuiteName:     "s",
		CreatedAt:     t0.Add(time.Second),
	}); err != nil {
		t.Fatalf("SaveSuiteResult: %v", err)
	}

	if _, err := st.GetVersionComparison(ctx, "p", "v1", "v2"); err == nil {
		t.Fatalf("GetVersionComparison: expected error")
	}
}

func TestSQLiteStore_LatestRunID_ScanError(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if err := st.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := st.latestRunID(ctx, "p", "v1"); err == nil || !strings.Contains(err.Error(), "latest run id") {
		t.Fatalf("latestRunID: %v", err)
	}
}

func TestSQLiteStore_SuiteResultsByRunPromptVersion_QueryError(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	if err := st.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := st.suiteResultsByRunPromptVersion(ctx, "run", "p", "v1"); err == nil {
		t.Fatalf("suiteResultsByRunPromptVersion: expected error")
	}
}

func TestScanSuiteRows_DecodeCaseResultsError(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()
	t0 := time.Unix(1_700_000_000, 0).UTC()

	if err := st.SaveRun(ctx, &RunRecord{ID: "run", StartedAt: t0, FinishedAt: t0.Add(time.Minute)}); err != nil {
		t.Fatalf("SaveRun: %v", err)
	}
	if _, err := st.db.ExecContext(ctx, `
		INSERT INTO suite_results (
			id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases, failed_cases,
			pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
		) VALUES ('suite_bad', 'run', 'p', 'v1', 's', 0, 0, 0, 0, 0, 0, 0, 1, ?)
	`, []byte("{")); err != nil {
		t.Fatalf("INSERT suite_bad: %v", err)
	}

	if _, err := st.GetSuiteResults(ctx, "run"); err == nil || !strings.Contains(err.Error(), "decode case results") {
		t.Fatalf("GetSuiteResults(decode): %v", err)
	}
}

func TestSQLiteStore_RowDecoders(t *testing.T) {
	if got, err := decodeConfig(sql.NullString{}); err != nil || got != nil {
		t.Fatalf("decodeConfig(null): got=%v err=%v", got, err)
	}
	if got, err := decodeConfig(sql.NullString{Valid: true, String: "null"}); err != nil || got != nil {
		t.Fatalf("decodeConfig(\"null\"): got=%v err=%v", got, err)
	}
	if _, err := decodeConfig(sql.NullString{Valid: true, String: "{"}); err == nil {
		t.Fatalf("decodeConfig(invalid): expected error")
	}

	if got, err := decodeCaseResults(nil); err != nil || got != nil {
		t.Fatalf("decodeCaseResults(nil): got=%v err=%v", got, err)
	}
	if _, err := decodeCaseResults([]byte("{")); err == nil {
		t.Fatalf("decodeCaseResults(invalid): expected error")
	}
}

func TestScanSuiteRows_ScanError(t *testing.T) {
	st := newTestSQLiteStore(t)

	rows, err := st.db.QueryContext(context.Background(), `SELECT 1`)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	defer rows.Close()

	if _, err := scanSuiteRows(rows); err == nil {
		t.Fatalf("scanSuiteRows: expected error")
	}
}

func TestCompareCaseOutcomes_IgnoreMissingCases(t *testing.T) {
	v1 := []*SuiteRecord{{
		CaseResults: []CaseRecord{
			{CaseID: "a", Passed: true},
			{CaseID: "b", Passed: false},
			{CaseID: "missing", Passed: true},
		},
	}}
	v2 := []*SuiteRecord{{
		CaseResults: []CaseRecord{
			{CaseID: "a", Passed: true},
			{CaseID: "b", Passed: false},
		},
	}}

	reg, imp := compareCaseOutcomes(v1, v2)
	if len(reg) != 0 || len(imp) != 0 {
		t.Fatalf("compareCaseOutcomes: regressions=%v improvements=%v", reg, imp)
	}
}

func TestSQLiteStore_LatestRunID_ErrorWrap(t *testing.T) {
	st := newTestSQLiteStore(t)
	ctx := context.Background()

	_, err := st.latestRunID(ctx, "p", "v1")
	if err == nil {
		t.Fatalf("latestRunID: expected error")
	}
	if !strings.Contains(err.Error(), "no runs") {
		t.Fatalf("latestRunID error: %v", err)
	}
}
