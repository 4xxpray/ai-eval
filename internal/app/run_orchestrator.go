package app

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

type SuiteRun struct {
	PromptName    string
	PromptVersion string
	Suite         *testcase.TestSuite
	Result        *runner.SuiteResult
}

type RunSummary struct {
	TotalSuites  int   `json:"total_suites"`
	TotalCases   int   `json:"total_cases"`
	PassedCases  int   `json:"passed_cases"`
	FailedCases  int   `json:"failed_cases"`
	TotalLatency int64 `json:"total_latency_ms"`
	TotalTokens  int   `json:"total_tokens"`
}

func IndexPrompts(prompts []*prompt.Prompt) (map[string]*prompt.Prompt, error) {
	out := make(map[string]*prompt.Prompt, len(prompts))
	for _, p := range prompts {
		if p == nil {
			return nil, fmt.Errorf("run: nil prompt")
		}
		name := strings.TrimSpace(p.Name)
		if name == "" {
			return nil, fmt.Errorf("run: prompt with empty name")
		}
		if _, ok := out[name]; ok {
			return nil, fmt.Errorf("run: duplicate prompt name %q", name)
		}
		out[name] = p
	}
	return out, nil
}

func IndexSuitesByPrompt(suites []*testcase.TestSuite, promptByName map[string]*prompt.Prompt) (map[string][]*testcase.TestSuite, error) {
	out := make(map[string][]*testcase.TestSuite)
	for _, s := range suites {
		if s == nil {
			return nil, fmt.Errorf("run: nil test suite")
		}
		promptRef := strings.TrimSpace(s.Prompt)
		if promptRef == "" {
			return nil, fmt.Errorf("run: suite %q: missing prompt reference", s.Suite)
		}
		if _, ok := promptByName[promptRef]; !ok {
			return nil, fmt.Errorf("run: suite %q references unknown prompt %q", s.Suite, promptRef)
		}
		out[promptRef] = append(out[promptRef], s)
	}
	return out, nil
}

func SummarizeRuns(runs []SuiteRun) (anyFailed bool, summary RunSummary) {
	summary.TotalSuites = len(runs)
	for _, r := range runs {
		if r.Result == nil {
			anyFailed = true
			continue
		}
		summary.TotalCases += r.Result.TotalCases
		summary.PassedCases += r.Result.PassedCases
		summary.FailedCases += r.Result.FailedCases
		summary.TotalLatency += r.Result.TotalLatency
		summary.TotalTokens += r.Result.TotalTokens
		if r.Result.FailedCases > 0 {
			anyFailed = true
		}
	}
	if summary.FailedCases > 0 {
		anyFailed = true
	}
	return anyFailed, summary
}

func SaveRun(ctx context.Context, writer store.RunWriter, runs []SuiteRun, summary RunSummary, startedAt, finishedAt time.Time, runConfig map[string]any) (*store.RunRecord, error) {
	if writer == nil {
		return nil, errors.New("run: missing store")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	runID, err := newRunID()
	if err != nil {
		return nil, fmt.Errorf("run: generate run id: %w", err)
	}

	passedSuites := 0
	failedSuites := 0
	for _, r := range runs {
		if r.Result != nil && r.Result.FailedCases == 0 {
			passedSuites++
		} else {
			failedSuites++
		}
	}

	runRecord := &store.RunRecord{
		ID:           runID,
		StartedAt:    startedAt,
		FinishedAt:   finishedAt,
		TotalSuites:  summary.TotalSuites,
		PassedSuites: passedSuites,
		FailedSuites: failedSuites,
		Config:       runConfig,
	}
	if err := writer.SaveRun(ctx, runRecord); err != nil {
		return nil, fmt.Errorf("run: save run: %w", err)
	}

	for i, r := range runs {
		if r.Result == nil || r.Suite == nil {
			return nil, errors.New("run: missing suite result")
		}

		caseResults := make([]store.CaseRecord, 0, len(r.Result.Results))
		for _, rr := range r.Result.Results {
			cr := store.CaseRecord{
				CaseID:     rr.CaseID,
				Passed:     rr.Passed,
				Score:      rr.Score,
				PassAtK:    rr.PassAtK,
				PassExpK:   rr.PassExpK,
				LatencyMs:  rr.LatencyMs,
				TokensUsed: rr.TokensUsed,
			}
			if rr.Error != nil {
				cr.Error = rr.Error.Error()
			}
			caseResults = append(caseResults, cr)
		}

		suiteRecord := &store.SuiteRecord{
			ID:            fmt.Sprintf("%s_suite_%d", runID, i+1),
			RunID:         runID,
			PromptName:    r.PromptName,
			PromptVersion: r.PromptVersion,
			SuiteName:     r.Suite.Suite,
			TotalCases:    r.Result.TotalCases,
			PassedCases:   r.Result.PassedCases,
			FailedCases:   r.Result.FailedCases,
			PassRate:      r.Result.PassRate,
			AvgScore:      r.Result.AvgScore,
			TotalLatency:  r.Result.TotalLatency,
			TotalTokens:   r.Result.TotalTokens,
			CreatedAt:     finishedAt,
			CaseResults:   caseResults,
		}
		if err := writer.SaveSuiteResult(ctx, suiteRecord); err != nil {
			return nil, fmt.Errorf("run: save suite result: %w", err)
		}
	}

	return runRecord, nil
}

func newRunID() (string, error) {
	var buf [8]byte
	if _, err := io.ReadFull(rand.Reader, buf[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("run_%s_%x", time.Now().UTC().Format("20060102T150405Z"), buf), nil
}
