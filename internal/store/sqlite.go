package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const defaultHistoryLimit = 50

// SQLiteStore implements Store using SQLite.
type SQLiteStore struct {
	db *sql.DB

	insertRunStmt                *sql.Stmt
	insertSuiteStmt              *sql.Stmt
	getRunStmt                   *sql.Stmt
	suitesByRunStmt              *sql.Stmt
	promptHistoryStmt            *sql.Stmt
	latestRunByPromptVersionStmt *sql.Stmt
	suitesByRunPromptVersionStmt *sql.Stmt
}

var (
	sqliteOpen              = sql.Open
	sqlitePrepareStatements = (*SQLiteStore).prepareStatements
)

// NewSQLiteStore opens or creates a SQLite store at the given path.
func NewSQLiteStore(path string) (*SQLiteStore, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, errors.New("store: empty sqlite path")
	}
	if path != ":memory:" {
		dir := filepath.Dir(path)
		if dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, fmt.Errorf("store: create sqlite dir: %w", err)
			}
		}
	}

	db, err := sqliteOpen("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("store: open sqlite: %w", err)
	}
	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)

	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("store: ping sqlite: %w", err)
	}

	if err := initSQLiteSchema(db); err != nil {
		_ = db.Close()
		return nil, err
	}

	st := &SQLiteStore{db: db}
	if err := sqlitePrepareStatements(st); err != nil {
		_ = st.Close()
		return nil, err
	}
	return st, nil
}

func initSQLiteSchema(db *sql.DB) error {
	stmts := []string{
		`PRAGMA foreign_keys = ON`,
		`PRAGMA journal_mode = WAL`,
		`CREATE TABLE IF NOT EXISTS runs (
			id TEXT PRIMARY KEY,
			started_at INTEGER NOT NULL,
			finished_at INTEGER NOT NULL,
			total_suites INTEGER NOT NULL,
			passed_suites INTEGER NOT NULL,
			failed_suites INTEGER NOT NULL,
			config_json TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS suite_results (
			id TEXT PRIMARY KEY,
			run_id TEXT NOT NULL,
			prompt_name TEXT NOT NULL,
			prompt_version TEXT NOT NULL,
			suite_name TEXT NOT NULL,
			total_cases INTEGER NOT NULL,
			passed_cases INTEGER NOT NULL,
			failed_cases INTEGER NOT NULL,
			pass_rate REAL NOT NULL,
			avg_score REAL NOT NULL,
			total_latency INTEGER NOT NULL,
			total_tokens INTEGER NOT NULL,
			created_at INTEGER NOT NULL,
			case_results BLOB NOT NULL,
			FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
		)`,
		`CREATE INDEX IF NOT EXISTS idx_suite_results_run_id ON suite_results(run_id)`,
		`CREATE INDEX IF NOT EXISTS idx_suite_results_prompt ON suite_results(prompt_name, prompt_version)`,
		`CREATE INDEX IF NOT EXISTS idx_suite_results_created_at ON suite_results(created_at)`,
	}

	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("store: init schema: %w", err)
		}
	}
	return nil
}

func (s *SQLiteStore) prepareStatements() error {
	if s == nil || s.db == nil {
		return errors.New("store: nil sqlite store")
	}

	ctx := context.Background()
	type stmtSpec struct {
		dst    **sql.Stmt
		query  string
		errFmt string
	}

	specs := []stmtSpec{
		{
			dst: &s.insertRunStmt,
			query: `
				INSERT INTO runs (
					id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json
				) VALUES (?, ?, ?, ?, ?, ?, ?)
			`,
			errFmt: "store: prepare insert run: %w",
		},
		{
			dst: &s.insertSuiteStmt,
			query: `
				INSERT INTO suite_results (
					id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases,
					failed_cases, pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			`,
			errFmt: "store: prepare insert suite: %w",
		},
		{
			dst: &s.getRunStmt,
			query: `
				SELECT id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json
				FROM runs WHERE id = ?
			`,
			errFmt: "store: prepare get run: %w",
		},
		{
			dst: &s.suitesByRunStmt,
			query: `
				SELECT id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases,
					failed_cases, pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
				FROM suite_results
				WHERE run_id = ?
				ORDER BY created_at ASC, suite_name ASC
			`,
			errFmt: "store: prepare get suites: %w",
		},
		{
			dst: &s.promptHistoryStmt,
			query: `
				SELECT id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases,
					failed_cases, pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
				FROM suite_results
				WHERE prompt_name = ?
				ORDER BY created_at DESC
				LIMIT ?
			`,
			errFmt: "store: prepare prompt history: %w",
		},
		{
			dst: &s.latestRunByPromptVersionStmt,
			query: `
				SELECT run_id FROM suite_results
				WHERE prompt_name = ? AND prompt_version = ?
				ORDER BY created_at DESC
				LIMIT 1
			`,
			errFmt: "store: prepare latest run: %w",
		},
		{
			dst: &s.suitesByRunPromptVersionStmt,
			query: `
				SELECT id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases,
					failed_cases, pass_rate, avg_score, total_latency, total_tokens, created_at, case_results
				FROM suite_results
				WHERE run_id = ? AND prompt_name = ? AND prompt_version = ?
				ORDER BY created_at ASC, suite_name ASC
			`,
			errFmt: "store: prepare suites by run/version: %w",
		},
	}

	for _, spec := range specs {
		stmt, err := s.db.PrepareContext(ctx, spec.query)
		if err != nil {
			return fmt.Errorf(spec.errFmt, err)
		}
		*spec.dst = stmt
	}

	return nil
}

// Close releases database resources.
func (s *SQLiteStore) Close() error {
	if s == nil {
		return nil
	}
	stmts := []*sql.Stmt{
		s.insertRunStmt,
		s.insertSuiteStmt,
		s.getRunStmt,
		s.suitesByRunStmt,
		s.promptHistoryStmt,
		s.latestRunByPromptVersionStmt,
		s.suitesByRunPromptVersionStmt,
	}
	for _, stmt := range stmts {
		if stmt != nil {
			_ = stmt.Close()
		}
	}
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// SaveRun persists a run summary.
func (s *SQLiteStore) SaveRun(ctx context.Context, run *RunRecord) error {
	if s == nil {
		return errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return errors.New("store: nil context")
	}
	if run == nil {
		return errors.New("store: nil run")
	}

	id := strings.TrimSpace(run.ID)
	if id == "" {
		return errors.New("store: empty run id")
	}
	if run.StartedAt.IsZero() || run.FinishedAt.IsZero() {
		return errors.New("store: missing run timestamps")
	}

	cfgJSON := []byte("null")
	if run.Config != nil {
		var err error
		cfgJSON, err = json.Marshal(run.Config)
		if err != nil {
			return fmt.Errorf("store: marshal run config: %w", err)
		}
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("store: begin run tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	stmt := tx.StmtContext(ctx, s.insertRunStmt)
	defer stmt.Close()

	_, err = stmt.ExecContext(
		ctx,
		id,
		run.StartedAt.UTC().UnixMilli(),
		run.FinishedAt.UTC().UnixMilli(),
		run.TotalSuites,
		run.PassedSuites,
		run.FailedSuites,
		string(cfgJSON),
	)
	if err != nil {
		return fmt.Errorf("store: insert run: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("store: commit run: %w", err)
	}
	return nil
}

// SaveSuiteResult persists a suite result.
func (s *SQLiteStore) SaveSuiteResult(ctx context.Context, result *SuiteRecord) error {
	if s == nil {
		return errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return errors.New("store: nil context")
	}
	if result == nil {
		return errors.New("store: nil suite result")
	}

	id := strings.TrimSpace(result.ID)
	if id == "" {
		return errors.New("store: empty suite id")
	}
	if strings.TrimSpace(result.RunID) == "" {
		return errors.New("store: empty run id")
	}
	if strings.TrimSpace(result.PromptName) == "" || strings.TrimSpace(result.SuiteName) == "" {
		return errors.New("store: missing prompt/suite name")
	}

	createdAt := result.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	caseJSON, err := json.Marshal(result.CaseResults)
	if err != nil {
		return fmt.Errorf("store: marshal case results: %w", err)
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("store: begin suite tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	stmt := tx.StmtContext(ctx, s.insertSuiteStmt)
	defer stmt.Close()

	_, err = stmt.ExecContext(
		ctx,
		id,
		result.RunID,
		result.PromptName,
		result.PromptVersion,
		result.SuiteName,
		result.TotalCases,
		result.PassedCases,
		result.FailedCases,
		result.PassRate,
		result.AvgScore,
		result.TotalLatency,
		result.TotalTokens,
		createdAt.UTC().UnixMilli(),
		caseJSON,
	)
	if err != nil {
		return fmt.Errorf("store: insert suite result: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("store: commit suite result: %w", err)
	}
	return nil
}

// GetRun loads a run by id.
func (s *SQLiteStore) GetRun(ctx context.Context, id string) (*RunRecord, error) {
	if s == nil {
		return nil, errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return nil, errors.New("store: nil context")
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return nil, errors.New("store: empty run id")
	}

	row := s.getRunStmt.QueryRowContext(ctx, id)
	var (
		runID        string
		startedAtMS  int64
		finishedAtMS int64
		totalSuites  int
		passedSuites int
		failedSuites int
		cfgJSON      sql.NullString
	)
	if err := row.Scan(&runID, &startedAtMS, &finishedAtMS, &totalSuites, &passedSuites, &failedSuites, &cfgJSON); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, err
		}
		return nil, fmt.Errorf("store: get run: %w", err)
	}

	cfg, err := decodeConfig(cfgJSON)
	if err != nil {
		return nil, fmt.Errorf("store: decode run config: %w", err)
	}

	return &RunRecord{
		ID:           runID,
		StartedAt:    time.UnixMilli(startedAtMS).UTC(),
		FinishedAt:   time.UnixMilli(finishedAtMS).UTC(),
		TotalSuites:  totalSuites,
		PassedSuites: passedSuites,
		FailedSuites: failedSuites,
		Config:       cfg,
	}, nil
}

// ListRuns returns runs matching the filter.
func (s *SQLiteStore) ListRuns(ctx context.Context, filter RunFilter) ([]*RunRecord, error) {
	if s == nil {
		return nil, errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return nil, errors.New("store: nil context")
	}

	promptName := strings.TrimSpace(filter.PromptName)
	promptVersion := strings.TrimSpace(filter.PromptVersion)
	limit := filter.Limit
	if limit <= 0 {
		limit = defaultHistoryLimit
	}

	var sb strings.Builder
	sb.WriteString(`SELECT DISTINCT r.id, r.started_at, r.finished_at, r.total_suites, r.passed_suites, r.failed_suites, r.config_json FROM runs r`)
	if promptName != "" || promptVersion != "" {
		sb.WriteString(` JOIN suite_results s ON s.run_id = r.id`)
	}
	sb.WriteString(` WHERE 1=1`)

	var args []any
	if promptName != "" {
		sb.WriteString(` AND s.prompt_name = ?`)
		args = append(args, promptName)
	}
	if promptVersion != "" {
		sb.WriteString(` AND s.prompt_version = ?`)
		args = append(args, promptVersion)
	}
	if !filter.Since.IsZero() {
		sb.WriteString(` AND r.started_at >= ?`)
		args = append(args, filter.Since.UTC().UnixMilli())
	}
	if !filter.Until.IsZero() {
		sb.WriteString(` AND r.started_at <= ?`)
		args = append(args, filter.Until.UTC().UnixMilli())
	}
	sb.WriteString(` ORDER BY r.started_at DESC`)
	if limit > 0 {
		sb.WriteString(` LIMIT ?`)
		args = append(args, limit)
	}

	rows, err := s.db.QueryContext(ctx, sb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("store: list runs: %w", err)
	}
	defer rows.Close()
	return scanRunRows(rows)
}

func scanRunRows(rows *sql.Rows) ([]*RunRecord, error) {
	var out []*RunRecord
	for rows.Next() {
		var (
			runID        string
			startedAtMS  int64
			finishedAtMS int64
			totalSuites  int
			passedSuites int
			failedSuites int
			cfgJSON      sql.NullString
		)
		if err := rows.Scan(&runID, &startedAtMS, &finishedAtMS, &totalSuites, &passedSuites, &failedSuites, &cfgJSON); err != nil {
			return nil, fmt.Errorf("store: scan run: %w", err)
		}
		cfg, err := decodeConfig(cfgJSON)
		if err != nil {
			return nil, fmt.Errorf("store: decode run config: %w", err)
		}
		out = append(out, &RunRecord{
			ID:           runID,
			StartedAt:    time.UnixMilli(startedAtMS).UTC(),
			FinishedAt:   time.UnixMilli(finishedAtMS).UTC(),
			TotalSuites:  totalSuites,
			PassedSuites: passedSuites,
			FailedSuites: failedSuites,
			Config:       cfg,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: list runs: %w", err)
	}
	return out, nil
}

// GetSuiteResults lists suite results for a run.
func (s *SQLiteStore) GetSuiteResults(ctx context.Context, runID string) ([]*SuiteRecord, error) {
	if s == nil {
		return nil, errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return nil, errors.New("store: nil context")
	}
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return nil, errors.New("store: empty run id")
	}

	rows, err := s.suitesByRunStmt.QueryContext(ctx, runID)
	if err != nil {
		return nil, fmt.Errorf("store: get suite results: %w", err)
	}
	defer rows.Close()

	return scanSuiteRows(rows)
}

// GetPromptHistory returns recent suite results for a prompt.
func (s *SQLiteStore) GetPromptHistory(ctx context.Context, promptName string, limit int) ([]*SuiteRecord, error) {
	if s == nil {
		return nil, errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return nil, errors.New("store: nil context")
	}
	promptName = strings.TrimSpace(promptName)
	if promptName == "" {
		return nil, errors.New("store: empty prompt name")
	}
	if limit <= 0 {
		limit = defaultHistoryLimit
	}

	rows, err := s.promptHistoryStmt.QueryContext(ctx, promptName, limit)
	if err != nil {
		return nil, fmt.Errorf("store: prompt history: %w", err)
	}
	defer rows.Close()

	return scanSuiteRows(rows)
}

// GetVersionComparison compares latest results between two prompt versions.
func (s *SQLiteStore) GetVersionComparison(ctx context.Context, promptName, v1, v2 string) (*VersionComparison, error) {
	if s == nil {
		return nil, errors.New("store: nil sqlite store")
	}
	if ctx == nil {
		return nil, errors.New("store: nil context")
	}
	promptName = strings.TrimSpace(promptName)
	v1 = strings.TrimSpace(v1)
	v2 = strings.TrimSpace(v2)
	if promptName == "" || v1 == "" || v2 == "" {
		return nil, errors.New("store: missing prompt/version")
	}

	runID1, err := s.latestRunID(ctx, promptName, v1)
	if err != nil {
		return nil, err
	}
	runID2, err := s.latestRunID(ctx, promptName, v2)
	if err != nil {
		return nil, err
	}

	runIDs := []string{runID1, runID2}
	versions := []string{v1, v2}
	results := make([][]*SuiteRecord, 2)
	for i := range runIDs {
		res, err := s.suiteResultsByRunPromptVersion(ctx, runIDs[i], promptName, versions[i])
		if err != nil {
			return nil, err
		}
		results[i] = res
	}

	regressions, improvements := compareCaseOutcomes(results[0], results[1])

	return &VersionComparison{
		PromptName:   promptName,
		V1:           v1,
		V2:           v2,
		V1Results:    results[0],
		V2Results:    results[1],
		Regressions:  regressions,
		Improvements: improvements,
	}, nil
}

func (s *SQLiteStore) latestRunID(ctx context.Context, promptName, version string) (string, error) {
	row := s.latestRunByPromptVersionStmt.QueryRowContext(ctx, promptName, version)
	var runID string
	if err := row.Scan(&runID); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", fmt.Errorf("store: no runs for prompt %q version %q", promptName, version)
		}
		return "", fmt.Errorf("store: latest run id: %w", err)
	}
	return runID, nil
}

func (s *SQLiteStore) suiteResultsByRunPromptVersion(ctx context.Context, runID, promptName, version string) ([]*SuiteRecord, error) {
	rows, err := s.suitesByRunPromptVersionStmt.QueryContext(ctx, runID, promptName, version)
	if err != nil {
		return nil, fmt.Errorf("store: suites by run/version: %w", err)
	}
	defer rows.Close()
	return scanSuiteRows(rows)
}

func scanSuiteRows(rows *sql.Rows) ([]*SuiteRecord, error) {
	var out []*SuiteRecord
	for rows.Next() {
		var (
			id            string
			runID         string
			promptName    string
			promptVersion string
			suiteName     string
			totalCases    int
			passedCases   int
			failedCases   int
			passRate      float64
			avgScore      float64
			totalLatency  int64
			totalTokens   int
			createdAtMS   int64
			caseJSON      []byte
		)
		if err := rows.Scan(
			&id,
			&runID,
			&promptName,
			&promptVersion,
			&suiteName,
			&totalCases,
			&passedCases,
			&failedCases,
			&passRate,
			&avgScore,
			&totalLatency,
			&totalTokens,
			&createdAtMS,
			&caseJSON,
		); err != nil {
			return nil, fmt.Errorf("store: scan suite: %w", err)
		}

		caseResults, err := decodeCaseResults(caseJSON)
		if err != nil {
			return nil, fmt.Errorf("store: decode case results: %w", err)
		}

		out = append(out, &SuiteRecord{
			ID:            id,
			RunID:         runID,
			PromptName:    promptName,
			PromptVersion: promptVersion,
			SuiteName:     suiteName,
			TotalCases:    totalCases,
			PassedCases:   passedCases,
			FailedCases:   failedCases,
			PassRate:      passRate,
			AvgScore:      avgScore,
			TotalLatency:  totalLatency,
			TotalTokens:   totalTokens,
			CreatedAt:     time.UnixMilli(createdAtMS).UTC(),
			CaseResults:   caseResults,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: scan suite rows: %w", err)
	}
	return out, nil
}

func decodeConfig(cfgJSON sql.NullString) (map[string]any, error) {
	if !cfgJSON.Valid {
		return nil, nil
	}
	raw := strings.TrimSpace(cfgJSON.String)
	if raw == "" || raw == "null" {
		return nil, nil
	}
	var cfg map[string]any
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func decodeCaseResults(caseJSON []byte) ([]CaseRecord, error) {
	if len(caseJSON) == 0 {
		return nil, nil
	}
	var out []CaseRecord
	if err := json.Unmarshal(caseJSON, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func compareCaseOutcomes(v1Results, v2Results []*SuiteRecord) ([]string, []string) {
	v1 := make(map[string]bool)
	for _, suite := range v1Results {
		for _, cr := range suite.CaseResults {
			v1[cr.CaseID] = cr.Passed
		}
	}
	v2 := make(map[string]bool)
	for _, suite := range v2Results {
		for _, cr := range suite.CaseResults {
			v2[cr.CaseID] = cr.Passed
		}
	}

	var regressions []string
	var improvements []string
	for caseID, v1Passed := range v1 {
		v2Passed, ok := v2[caseID]
		if !ok {
			continue
		}
		if v1Passed && !v2Passed {
			regressions = append(regressions, caseID)
		}
		if !v1Passed && v2Passed {
			improvements = append(improvements, caseID)
		}
	}

	sort.Strings(regressions)
	sort.Strings(improvements)
	return regressions, improvements
}
