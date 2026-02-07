package runner

import (
	"time"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// Config defines runner behavior and thresholds.
type Config struct {
	Trials        int     // Number of trials for non-deterministic evaluation
	PassThreshold float64 // Threshold for pass@k
	Concurrency   int     // Max concurrent evaluations
	Timeout       time.Duration
}

// RunResult reports results for a single test case.
type RunResult struct {
	Suite      string
	CaseID     string
	Passed     bool
	Score      float64
	Trials     []TrialResult
	PassAtK    float64 // At least one pass in k trials
	PassExpK   float64 // All k trials pass
	LatencyMs  int64
	TokensUsed int
	Error      error
}

// TrialResult reports the outcome of a single trial.
type TrialResult struct {
	TrialNum    int
	Response    string
	ToolCalls   []llm.ToolUse
	Evaluations []evaluator.Result
	Passed      bool
	Score       float64
	LatencyMs   int64
}

// SuiteResult aggregates results for a test suite.
type SuiteResult struct {
	Suite        string
	TotalCases   int
	PassedCases  int
	FailedCases  int
	PassRate     float64
	AvgScore     float64
	TotalLatency int64
	TotalTokens  int
	Results      []RunResult
}
