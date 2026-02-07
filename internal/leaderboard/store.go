package leaderboard

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const defaultLimit = 50

type Store struct {
	db *sql.DB
}

type Entry struct {
	ID       int64
	Model    string
	Provider string
	Dataset  string
	Score    float64
	Accuracy float64
	Latency  int64
	Cost     float64
	EvalDate time.Time
}

func NewStore(dbPath string) (*Store, error) {
	dbPath = strings.TrimSpace(dbPath)
	if dbPath == "" {
		return nil, errors.New("leaderboard: empty db path")
	}

	if dbPath != ":memory:" {
		dir := filepath.Dir(dbPath)
		if dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, fmt.Errorf("leaderboard: create db dir: %w", err)
			}
		}
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("leaderboard: open db: %w", err)
	}
	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)

	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("leaderboard: ping db: %w", err)
	}
	if err := initSchema(db); err != nil {
		_ = db.Close()
		return nil, err
	}

	return &Store{db: db}, nil
}

func initSchema(db *sql.DB) error {
	if db == nil {
		return errors.New("leaderboard: nil db")
	}

	stmts := []string{
		`PRAGMA foreign_keys = ON`,
		`PRAGMA journal_mode = WAL`,
		`CREATE TABLE IF NOT EXISTS leaderboard_entries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			model TEXT NOT NULL,
			provider TEXT NOT NULL,
			dataset TEXT NOT NULL,
			score REAL NOT NULL,
			accuracy REAL NOT NULL,
			latency INTEGER NOT NULL,
			cost REAL NOT NULL,
			eval_date INTEGER NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_leaderboard_dataset ON leaderboard_entries(dataset)`,
		`CREATE INDEX IF NOT EXISTS idx_leaderboard_model_dataset ON leaderboard_entries(model, dataset)`,
		`CREATE INDEX IF NOT EXISTS idx_leaderboard_eval_date ON leaderboard_entries(eval_date)`,
	}

	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("leaderboard: init schema: %w", err)
		}
	}
	return nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) Save(ctx context.Context, entry *Entry) error {
	if s == nil || s.db == nil {
		return errors.New("leaderboard: nil store")
	}
	if ctx == nil {
		return errors.New("leaderboard: nil context")
	}
	if entry == nil {
		return errors.New("leaderboard: nil entry")
	}

	model := strings.TrimSpace(entry.Model)
	provider := strings.TrimSpace(entry.Provider)
	dataset := strings.TrimSpace(entry.Dataset)
	if model == "" || provider == "" || dataset == "" {
		return errors.New("leaderboard: missing model/provider/dataset")
	}

	evalDate := entry.EvalDate
	if evalDate.IsZero() {
		evalDate = time.Now().UTC()
	}

	res, err := s.db.ExecContext(ctx, `
		INSERT INTO leaderboard_entries (
			model, provider, dataset, score, accuracy, latency, cost, eval_date
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, model, provider, dataset, entry.Score, entry.Accuracy, entry.Latency, entry.Cost, evalDate.UTC().UnixMilli())
	if err != nil {
		return fmt.Errorf("leaderboard: insert entry: %w", err)
	}

	if id, err := res.LastInsertId(); err == nil {
		entry.ID = id
	}
	entry.EvalDate = evalDate
	entry.Model = model
	entry.Provider = provider
	entry.Dataset = dataset
	return nil
}

func (s *Store) GetLeaderboard(ctx context.Context, dataset string, limit int) ([]Entry, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("leaderboard: nil store")
	}
	if ctx == nil {
		return nil, errors.New("leaderboard: nil context")
	}
	dataset = strings.TrimSpace(dataset)
	if dataset == "" {
		return nil, errors.New("leaderboard: empty dataset")
	}
	if limit <= 0 {
		limit = defaultLimit
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, model, provider, dataset, score, accuracy, latency, cost, eval_date
		FROM leaderboard_entries
		WHERE dataset = ?
		ORDER BY score DESC, accuracy DESC, latency ASC, eval_date DESC
		LIMIT ?
	`, dataset, limit)
	if err != nil {
		return nil, fmt.Errorf("leaderboard: query leaderboard: %w", err)
	}
	defer rows.Close()

	return scanRows(rows)
}

func (s *Store) GetModelHistory(ctx context.Context, model, dataset string) ([]Entry, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("leaderboard: nil store")
	}
	if ctx == nil {
		return nil, errors.New("leaderboard: nil context")
	}
	model = strings.TrimSpace(model)
	dataset = strings.TrimSpace(dataset)
	if model == "" || dataset == "" {
		return nil, errors.New("leaderboard: missing model/dataset")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, model, provider, dataset, score, accuracy, latency, cost, eval_date
		FROM leaderboard_entries
		WHERE model = ? AND dataset = ?
		ORDER BY eval_date DESC
	`, model, dataset)
	if err != nil {
		return nil, fmt.Errorf("leaderboard: query model history: %w", err)
	}
	defer rows.Close()

	return scanRows(rows)
}

func scanRows(rows *sql.Rows) ([]Entry, error) {
	var out []Entry
	for rows.Next() {
		var e Entry
		var evalDateMS int64
		if err := rows.Scan(
			&e.ID,
			&e.Model,
			&e.Provider,
			&e.Dataset,
			&e.Score,
			&e.Accuracy,
			&e.Latency,
			&e.Cost,
			&evalDateMS,
		); err != nil {
			return nil, fmt.Errorf("leaderboard: scan entry: %w", err)
		}
		e.EvalDate = time.UnixMilli(evalDateMS).UTC()
		out = append(out, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("leaderboard: scan rows: %w", err)
	}
	return out, nil
}

