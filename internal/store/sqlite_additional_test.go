package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	sqlite3 "github.com/mattn/go-sqlite3"
)

func TestNewSQLiteStore_OpenError(t *testing.T) {
	old := sqliteOpen
	sqliteOpen = func(driverName, dataSourceName string) (*sql.DB, error) {
		_ = driverName
		_ = dataSourceName
		return nil, errors.New("boom")
	}
	t.Cleanup(func() { sqliteOpen = old })

	if _, err := NewSQLiteStore(":memory:"); err == nil {
		t.Fatalf("NewSQLiteStore(open): expected error")
	}
}

func TestNewSQLiteStore_PrepareStatementsError(t *testing.T) {
	old := sqlitePrepareStatements
	sqlitePrepareStatements = func(*SQLiteStore) error {
		return errors.New("boom")
	}
	t.Cleanup(func() { sqlitePrepareStatements = old })

	if _, err := NewSQLiteStore(":memory:"); err == nil {
		t.Fatalf("NewSQLiteStore(prepareStatements): expected error")
	}
}

func TestSQLiteStore_prepareStatements_ClosedDB(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	st := &SQLiteStore{db: db}
	if err := st.prepareStatements(); err == nil {
		t.Fatalf("prepareStatements(closed db): expected error")
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

func TestScanRunRows_RowsErr(t *testing.T) {
	driverName := registerBoomDriver(t)
	db, err := sql.Open(driverName, ":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	if err := initSQLiteSchema(db); err != nil {
		t.Fatalf("initSQLiteSchema: %v", err)
	}

	if _, err := db.ExecContext(context.Background(), `
		INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json)
		VALUES
			('r1', 1, 2, 0, 0, 0, NULL),
			('r2', 3, 4, 0, 0, 0, NULL)
	`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}

	rows, err := db.QueryContext(context.Background(), `
		SELECT id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json
		FROM runs
		WHERE boom(started_at) = 1
		ORDER BY started_at DESC
	`)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	defer rows.Close()

	if _, err := scanRunRows(rows); err == nil || !strings.Contains(err.Error(), "list runs") {
		t.Fatalf("scanRunRows: %v", err)
	}
}

func TestScanSuiteRows_RowsErr(t *testing.T) {
	driverName := registerBoomDriver(t)
	db, err := sql.Open(driverName, ":memory:")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	if err := initSQLiteSchema(db); err != nil {
		t.Fatalf("initSQLiteSchema: %v", err)
	}

	if _, err := db.ExecContext(context.Background(), `
		INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json)
		VALUES ('run', 1, 2, 0, 0, 0, NULL)
	`); err != nil {
		t.Fatalf("INSERT run: %v", err)
	}
	if _, err := db.ExecContext(context.Background(), `
		INSERT INTO suite_results (
			id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases, failed_cases,
			pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
		) VALUES
			('s1', 'run', 'p', 'v', 'suite', 0, 0, 0, 0, 0, 0, 0, 1, X'5B5D'),
			('s2', 'run', 'p', 'v', 'suite', 0, 0, 0, 0, 0, 0, 0, 2, X'5B5D')
	`); err != nil {
		t.Fatalf("INSERT suites: %v", err)
	}

	rows, err := db.QueryContext(context.Background(), `
		SELECT id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases,
			failed_cases, pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
		FROM suite_results
		WHERE boom(created_at) = 1
		ORDER BY created_at ASC
	`)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	defer rows.Close()

	if _, err := scanSuiteRows(rows); err == nil || !strings.Contains(err.Error(), "scan suite rows") {
		t.Fatalf("scanSuiteRows: %v", err)
	}
}
