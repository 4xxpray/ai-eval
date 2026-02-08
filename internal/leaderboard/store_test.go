package leaderboard

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sqlite3 "github.com/mattn/go-sqlite3"
)

func TestNewStore_ValidationAndSetup(t *testing.T) {
	if _, err := NewStore("  "); err == nil {
		t.Fatalf("NewStore: expected error for empty path")
	}

	if err := initSchema(nil); err == nil {
		t.Fatalf("initSchema: expected error for nil db")
	}

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "nested", "leaderboard.db")
	st, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	if _, err := os.Stat(filepath.Dir(dbPath)); err != nil {
		t.Fatalf("db dir: %v", err)
	}
}

func TestNewStore_OpenError(t *testing.T) {
	old := sqlOpen
	sqlOpen = func(driverName, dataSourceName string) (*sql.DB, error) {
		_ = driverName
		_ = dataSourceName
		return nil, errors.New("boom")
	}
	t.Cleanup(func() { sqlOpen = old })

	if _, err := NewStore(":memory:"); err == nil {
		t.Fatalf("NewStore(open): expected error")
	}
}

func TestNewStore_Errors(t *testing.T) {
	dir := t.TempDir()

	notADir := filepath.Join(dir, "notadir")
	if err := os.WriteFile(notADir, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := NewStore(filepath.Join(notADir, "lb.db")); err == nil {
		t.Fatalf("NewStore(mkdir): expected error")
	}

	if _, err := NewStore(dir); err == nil {
		t.Fatalf("NewStore(ping): expected error")
	}
}

func TestNewStore_InitSchemaError_ReadOnlyDSN(t *testing.T) {
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

	if _, err := NewStore("file:ro.db?mode=ro"); err == nil {
		t.Fatalf("NewStore(read-only): expected error")
	}
}

func TestNewStore_NoDir_Succeeds(t *testing.T) {
	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	tmp := t.TempDir()
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWd) })

	st, err := NewStore("lb.db")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	if _, err := os.Stat(filepath.Join(tmp, "lb.db")); err != nil {
		t.Fatalf("stat: %v", err)
	}
}

func TestInitSchema_ClosedDB(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := initSchema(db); err == nil {
		t.Fatalf("initSchema: expected error for closed db")
	}
}

func TestStore_SaveAndGetLeaderboard(t *testing.T) {
	st, err := NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer st.Close()

	ctx := context.Background()
	e1 := &Entry{
		Model:    "m1",
		Provider: "openai",
		Dataset:  "mmlu",
		Score:    0.80,
		Accuracy: 0.80,
		Latency:  120,
		Cost:     0,
		EvalDate: time.UnixMilli(1000).UTC(),
	}
	e2 := &Entry{
		Model:    "m2",
		Provider: "openai",
		Dataset:  "mmlu",
		Score:    0.90,
		Accuracy: 0.90,
		Latency:  200,
		Cost:     0,
		EvalDate: time.UnixMilli(2000).UTC(),
	}

	if err := st.Save(ctx, e1); err != nil {
		t.Fatalf("Save e1: %v", err)
	}
	if err := st.Save(ctx, e2); err != nil {
		t.Fatalf("Save e2: %v", err)
	}
	if e1.ID == 0 || e2.ID == 0 {
		t.Fatalf("expected IDs to be set (got e1=%d e2=%d)", e1.ID, e2.ID)
	}

	got, err := st.GetLeaderboard(ctx, "mmlu", 10)
	if err != nil {
		t.Fatalf("GetLeaderboard: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(entries): got %d want %d", len(got), 2)
	}
	if got[0].Model != "m2" {
		t.Fatalf("rank1 model: got %q want %q", got[0].Model, "m2")
	}
	if got[1].Model != "m1" {
		t.Fatalf("rank2 model: got %q want %q", got[1].Model, "m1")
	}
}

func TestStore_GetModelHistory_Order(t *testing.T) {
	st, err := NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer st.Close()

	ctx := context.Background()
	if err := st.Save(ctx, &Entry{
		Model:    "m1",
		Provider: "claude",
		Dataset:  "gsm8k",
		Score:    0.20,
		Accuracy: 0.20,
		Latency:  10,
		Cost:     0,
		EvalDate: time.UnixMilli(1000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := st.Save(ctx, &Entry{
		Model:    "m1",
		Provider: "claude",
		Dataset:  "gsm8k",
		Score:    0.90,
		Accuracy: 0.90,
		Latency:  20,
		Cost:     0,
		EvalDate: time.UnixMilli(2000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := st.GetModelHistory(ctx, "m1", "gsm8k")
	if err != nil {
		t.Fatalf("GetModelHistory: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(history): got %d want %d", len(got), 2)
	}
	if got[0].Score != 0.90 {
		t.Fatalf("history[0].Score: got %.2f want %.2f", got[0].Score, 0.90)
	}
	if got[1].Score != 0.20 {
		t.Fatalf("history[1].Score: got %.2f want %.2f", got[1].Score, 0.20)
	}
}

func TestStore_ErrorsAndDefaults(t *testing.T) {
	if err := (*Store)(nil).Close(); err != nil {
		t.Fatalf("Close(nil): %v", err)
	}
	if err := (&Store{}).Close(); err != nil {
		t.Fatalf("Close(nil db): %v", err)
	}

	st, err := NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	ctx := context.Background()

	if err := (*Store)(nil).Save(ctx, &Entry{Model: "m", Provider: "p", Dataset: "d"}); err == nil {
		t.Fatalf("Save(nil store): expected error")
	}
	if err := st.Save(nil, &Entry{Model: "m", Provider: "p", Dataset: "d"}); err == nil {
		t.Fatalf("Save(nil ctx): expected error")
	}
	if err := st.Save(ctx, nil); err == nil {
		t.Fatalf("Save(nil entry): expected error")
	}
	if err := st.Save(ctx, &Entry{Model: " ", Provider: "p", Dataset: "d"}); err == nil {
		t.Fatalf("Save(missing fields): expected error")
	}

	e := &Entry{
		Model:    " m ",
		Provider: " p ",
		Dataset:  " d ",
		Score:    1,
		Accuracy: 1,
	}
	if err := st.Save(ctx, e); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if e.Model != "m" || e.Provider != "p" || e.Dataset != "d" {
		t.Fatalf("trim: got %#v", e)
	}
	if e.EvalDate.IsZero() {
		t.Fatalf("EvalDate: expected non-zero")
	}

	if _, err := (*Store)(nil).GetLeaderboard(ctx, "d", 1); err == nil {
		t.Fatalf("GetLeaderboard(nil store): expected error")
	}
	if _, err := st.GetLeaderboard(nil, "d", 1); err == nil {
		t.Fatalf("GetLeaderboard(nil ctx): expected error")
	}
	if _, err := st.GetLeaderboard(ctx, "  ", 1); err == nil {
		t.Fatalf("GetLeaderboard(empty dataset): expected error")
	}
	if _, err := st.GetLeaderboard(ctx, "d", 0); err != nil {
		t.Fatalf("GetLeaderboard(default limit): %v", err)
	}

	if _, err := (*Store)(nil).GetModelHistory(ctx, "m", "d"); err == nil {
		t.Fatalf("GetModelHistory(nil store): expected error")
	}
	if _, err := st.GetModelHistory(nil, "m", "d"); err == nil {
		t.Fatalf("GetModelHistory(nil ctx): expected error")
	}
	if _, err := st.GetModelHistory(ctx, "  ", "d"); err == nil {
		t.Fatalf("GetModelHistory(missing): expected error")
	}
	if _, err := st.GetModelHistory(ctx, "m", "  "); err == nil {
		t.Fatalf("GetModelHistory(missing dataset): expected error")
	}

	if err := st.db.Close(); err != nil {
		t.Fatalf("Close db: %v", err)
	}
	if err := st.Save(ctx, &Entry{Model: "m", Provider: "p", Dataset: "d"}); err == nil {
		t.Fatalf("Save(closed db): expected error")
	}
	if _, err := st.GetLeaderboard(ctx, "d", 1); err == nil {
		t.Fatalf("GetLeaderboard(closed db): expected error")
	}
	if _, err := st.GetModelHistory(ctx, "m", "d"); err == nil {
		t.Fatalf("GetModelHistory(closed db): expected error")
	}
}

func TestScanRows_ScanError(t *testing.T) {
	st, err := NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	rows, err := st.db.QueryContext(context.Background(), `SELECT 1`)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	defer rows.Close()

	if _, err := scanRows(rows); err == nil {
		t.Fatalf("scanRows: expected error")
	}
}

var boomDriverCounter int64

func registerBoomDriver(t *testing.T) string {
	t.Helper()

	driverName := fmt.Sprintf("sqlite3_boom_%d", atomic.AddInt64(&boomDriverCounter, 1))
	sql.Register(driverName, &sqlite3.SQLiteDriver{
		ConnectHook: func(conn *sqlite3.SQLiteConn) error {
			var calls int32
			return conn.RegisterFunc("boom", func(_ any) (int, error) {
				if atomic.AddInt32(&calls, 1) >= 2 {
					return 0, errors.New("boom")
				}
				return 1, nil
			}, false)
		},
	})
	return driverName
}

func TestScanRows_RowsErr(t *testing.T) {
	driverName := registerBoomDriver(t)
	db, err := sql.Open(driverName, ":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	if err := initSchema(db); err != nil {
		t.Fatalf("initSchema: %v", err)
	}
	if _, err := db.ExecContext(context.Background(), `
		INSERT INTO leaderboard_entries (model, provider, dataset, score, accuracy, latency, cost, eval_date)
		VALUES
			('m1', 'p', 'd', 1, 1, 1, 0, 1),
			('m2', 'p', 'd', 1, 1, 1, 0, 2)
	`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}

	rows, err := db.QueryContext(context.Background(), `
		SELECT id, model, provider, dataset, score, accuracy, latency, cost, eval_date
		FROM leaderboard_entries
		WHERE boom(eval_date) = 1
		ORDER BY eval_date DESC
	`)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	defer rows.Close()

	if _, err := scanRows(rows); err == nil || !strings.Contains(err.Error(), "scan rows") {
		t.Fatalf("scanRows: %v", err)
	}
}
