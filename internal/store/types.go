package store

import (
	"context"
	"time"
)

// RunWriter defines persistence for run summaries and suite results.
type RunWriter interface {
	// Save evaluation results
	SaveRun(ctx context.Context, run *RunRecord) error
	SaveSuiteResult(ctx context.Context, result *SuiteRecord) error
}

// RunReader defines read access to run and suite data.
type RunReader interface {
	// Query results
	GetRun(ctx context.Context, id string) (*RunRecord, error)
	ListRuns(ctx context.Context, filter RunFilter) ([]*RunRecord, error)
	GetSuiteResults(ctx context.Context, runID string) ([]*SuiteRecord, error)
}

// Analytics defines query helpers for historical comparisons.
type Analytics interface {
	// Analytics
	GetPromptHistory(ctx context.Context, promptName string, limit int) ([]*SuiteRecord, error)
	GetVersionComparison(ctx context.Context, promptName, v1, v2 string) (*VersionComparison, error)
}

// Store defines persistence for runs and suite results.
type Store interface {
	RunWriter
	RunReader
	Analytics
	Close() error
}

// RunRecord stores a single run summary.
type RunRecord struct {
	ID           string
	StartedAt    time.Time
	FinishedAt   time.Time
	TotalSuites  int
	PassedSuites int
	FailedSuites int
	Config       map[string]any // Serialized config
}

// SuiteRecord stores results for one prompt and suite.
type SuiteRecord struct {
	ID            string
	RunID         string
	PromptName    string
	PromptVersion string
	SuiteName     string
	TotalCases    int
	PassedCases   int
	FailedCases   int
	PassRate      float64
	AvgScore      float64
	TotalLatency  int64
	TotalTokens   int
	CreatedAt     time.Time
	CaseResults   []CaseRecord // JSON serialized
}

// CaseRecord stores a single test case result.
type CaseRecord struct {
	CaseID     string
	Passed     bool
	Score      float64
	PassAtK    float64
	PassExpK   float64
	LatencyMs  int64
	TokensUsed int
	Error      string
}

// RunFilter filters run listings.
type RunFilter struct {
	PromptName    string
	PromptVersion string
	Since         time.Time
	Until         time.Time
	Limit         int
}

// VersionComparison summarizes regressions between prompt versions.
type VersionComparison struct {
	PromptName   string
	V1           string
	V2           string
	V1Results    []*SuiteRecord
	V2Results    []*SuiteRecord
	Regressions  []string // Case IDs that regressed
	Improvements []string // Case IDs that improved
}
