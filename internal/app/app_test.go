package app

import (
	"context"
	"crypto/rand"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestLoadPromptsAndSuites(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "a.yaml"), `
name: p1
version: v1
template: hello
`)
	writeFile(t, filepath.Join(dir, "b.yml"), `
name: p2
version: v2
template: hello
`)
	writeFile(t, filepath.Join(dir, "c.txt"), `ignored`)

	ps, err := LoadPrompts(dir)
	if err != nil {
		t.Fatalf("LoadPrompts: %v", err)
	}
	if len(ps) != 2 {
		t.Fatalf("len(prompts): got %d want %d", len(ps), 2)
	}

	suiteDir := filepath.Join(dir, "suites")
	if err := os.MkdirAll(suiteDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	writeFile(t, filepath.Join(suiteDir, "s1.yaml"), `
suite: s1
prompt: p1
cases:
  - id: c1
    input: {}
    expected:
      contains: ["x"]
`)

	ss, err := LoadTestSuites(suiteDir)
	if err != nil {
		t.Fatalf("LoadTestSuites: %v", err)
	}
	if len(ss) != 1 || ss[0].Suite != "s1" {
		t.Fatalf("suites: %#v", ss)
	}
}

func TestLoadPromptsRecursive(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	writeFile(t, filepath.Join(sub, "b.yaml"), `
name: p2
version: v2
template: hello
`)
	writeFile(t, filepath.Join(sub, "ignored.txt"), `x`)
	writeFile(t, filepath.Join(sub, "c.yml"), `
name: p3
version: v3
template: hello
`)
	writeFile(t, filepath.Join(dir, "a.yaml"), `
name: p1
version: v1
template: hello
`)

	ps, err := LoadPromptsRecursive(dir)
	if err != nil {
		t.Fatalf("LoadPromptsRecursive: %v", err)
	}
	if len(ps) != 3 {
		t.Fatalf("len(prompts): got %d want %d", len(ps), 3)
	}
	if ps[0].Name != "p1" || ps[1].Name != "p2" || ps[2].Name != "p3" {
		t.Fatalf("order: %#v", []string{ps[0].Name, ps[1].Name, ps[2].Name})
	}

	_, err = LoadPromptsRecursive(filepath.Join(dir, "missing"))
	if err == nil || !strings.Contains(err.Error(), "prompt: walk dir") {
		t.Fatalf("LoadPromptsRecursive(missing): got %v", err)
	}

	writeFile(t, filepath.Join(dir, "bad.yaml"), ":")
	_, err = LoadPromptsRecursive(dir)
	if err == nil || !strings.Contains(err.Error(), "prompt: parse") {
		t.Fatalf("LoadPromptsRecursive(bad yaml): got %v", err)
	}
}

func TestFindPromptHelpers(t *testing.T) {
	ps := []*prompt.Prompt{
		{Name: "p", Version: "v1.2"},
		{Name: "p", Version: "v1.10"},
		{Name: "q", Version: "beta"},
		nil,
	}

	_, err := FindPromptByNameVersion(ps, " ", "v1")
	if err == nil {
		t.Fatalf("FindPromptByNameVersion: expected error")
	}

	p, err := FindPromptByNameVersion(ps, "p", "v1.10")
	if err != nil {
		t.Fatalf("FindPromptByNameVersion: %v", err)
	}
	if p == nil || p.Version != "v1.10" {
		t.Fatalf("prompt: %#v", p)
	}

	_, err = FindPromptByNameVersion([]*prompt.Prompt{{Name: "p", Version: "v1"}, {Name: "p", Version: "v1"}}, "p", "v1")
	if err == nil || !strings.Contains(err.Error(), "multiple matches") {
		t.Fatalf("FindPromptByNameVersion(dup): got %v", err)
	}
	_, err = FindPromptByNameVersion(ps, "p", "v9")
	if err == nil || !strings.Contains(err.Error(), "no prompt found") {
		t.Fatalf("FindPromptByNameVersion(miss): got %v", err)
	}

	latest, err := FindPromptLatestByName(ps, "p")
	if err != nil {
		t.Fatalf("FindPromptLatestByName: %v", err)
	}
	if latest.Version != "v1.10" {
		t.Fatalf("latest: got %q want %q", latest.Version, "v1.10")
	}

	_, err = FindPromptLatestByName(ps, " ")
	if err == nil {
		t.Fatalf("FindPromptLatestByName(empty): expected error")
	}
	_, err = FindPromptLatestByName(ps, "missing")
	if err == nil || !strings.Contains(err.Error(), "unknown prompt") {
		t.Fatalf("FindPromptLatestByName(missing): got %v", err)
	}

	if _, ok := parseNumericVersion("v1.2.3"); !ok {
		t.Fatalf("parseNumericVersion(v1.2.3): expected ok")
	}
	if _, ok := parseNumericVersion("v1..2"); ok {
		t.Fatalf("parseNumericVersion(v1..2): expected not ok")
	}
	if _, ok := parseNumericVersion("v1.x"); ok {
		t.Fatalf("parseNumericVersion(v1.x): expected not ok")
	}
	if _, ok := parseNumericVersion("  "); ok {
		t.Fatalf("parseNumericVersion(empty): expected not ok")
	}
	if got := compareVersions("beta", "alpha"); got <= 0 {
		t.Fatalf("compareVersions(beta,alpha): got %d want >0", got)
	}
}

func TestCompareVersions(t *testing.T) {
	if got := compareVersions("v1.2.0", "1.2"); got != 0 {
		t.Fatalf("compareVersions(v1.2.0,1.2): got %d want %d", got, 0)
	}
	if got := compareVersions("v1.2.1", "1.2"); got <= 0 {
		t.Fatalf("compareVersions(v1.2.1,1.2): got %d want >0", got)
	}
	if got := compareVersions("v1.2", "1.2.3"); got >= 0 {
		t.Fatalf("compareVersions(v1.2,1.2.3): got %d want <0", got)
	}
	if got := compareVersions("1.2.0", "1.2.0"); got != 0 {
		t.Fatalf("compareVersions(equal): got %d want %d", got, 0)
	}
}

func TestFindPromptLatestByName_DuplicateVersion(t *testing.T) {
	_, err := FindPromptLatestByName([]*prompt.Prompt{
		{Name: "p", Version: "v1"},
		{Name: "p", Version: "1"},
	}, "p")
	if err == nil || !strings.Contains(err.Error(), "multiple matches") {
		t.Fatalf("FindPromptLatestByName(dup): got %v", err)
	}
}

func TestFilterIndexAndSummaries(t *testing.T) {
	suites := []*testcase.TestSuite{
		nil,
		{Suite: "s1", Prompt: "p1"},
		{Suite: "s2", Prompt: "p2"},
		{Suite: "s3", Prompt: " p1 "},
	}
	if got := FilterSuitesByPrompt(suites, ""); got != nil {
		t.Fatalf("FilterSuitesByPrompt(empty): %#v", got)
	}
	got := FilterSuitesByPrompt(suites, "p1")
	if len(got) != 2 || got[0].Suite != "s1" || got[1].Suite != "s3" {
		t.Fatalf("FilterSuitesByPrompt(p1): %#v", got)
	}

	_, err := IndexPrompts([]*prompt.Prompt{nil})
	if err == nil {
		t.Fatalf("IndexPrompts(nil prompt): expected error")
	}
	_, err = IndexPrompts([]*prompt.Prompt{{Name: " "}})
	if err == nil {
		t.Fatalf("IndexPrompts(empty name): expected error")
	}
	_, err = IndexPrompts([]*prompt.Prompt{{Name: "p"}, {Name: "p"}})
	if err == nil {
		t.Fatalf("IndexPrompts(dup): expected error")
	}
	promptByName, err := IndexPrompts([]*prompt.Prompt{{Name: "p1"}, {Name: "p2"}})
	if err != nil {
		t.Fatalf("IndexPrompts: %v", err)
	}

	_, err = IndexSuitesByPrompt([]*testcase.TestSuite{nil}, promptByName)
	if err == nil {
		t.Fatalf("IndexSuitesByPrompt(nil suite): expected error")
	}
	_, err = IndexSuitesByPrompt([]*testcase.TestSuite{{Suite: "s", Prompt: " "}}, promptByName)
	if err == nil {
		t.Fatalf("IndexSuitesByPrompt(missing ref): expected error")
	}
	_, err = IndexSuitesByPrompt([]*testcase.TestSuite{{Suite: "s", Prompt: "unknown"}}, promptByName)
	if err == nil {
		t.Fatalf("IndexSuitesByPrompt(unknown ref): expected error")
	}

	byPrompt, err := IndexSuitesByPrompt([]*testcase.TestSuite{{Suite: "s1", Prompt: "p1"}, {Suite: "s2", Prompt: "p1"}}, promptByName)
	if err != nil {
		t.Fatalf("IndexSuitesByPrompt: %v", err)
	}
	if len(byPrompt["p1"]) != 2 {
		t.Fatalf("byPrompt[p1]: %#v", byPrompt["p1"])
	}

	anyFailed, summary := SummarizeRuns([]SuiteRun{
		{Result: nil},
		{Result: &runner.SuiteResult{TotalCases: 2, PassedCases: 2, FailedCases: 0, TotalLatency: 3, TotalTokens: 4}},
		{Result: &runner.SuiteResult{TotalCases: 1, PassedCases: 0, FailedCases: 1}},
	})
	if !anyFailed {
		t.Fatalf("anyFailed: got false want true")
	}
	if summary.TotalSuites != 3 || summary.TotalCases != 3 || summary.FailedCases != 1 || summary.TotalTokens != 4 {
		t.Fatalf("summary: %#v", summary)
	}

	anyFailed, summary = SummarizeRuns(nil)
	if anyFailed || summary.TotalSuites != 0 {
		t.Fatalf("SummarizeRuns(nil): anyFailed=%v summary=%#v", anyFailed, summary)
	}
}

func TestSaveRun(t *testing.T) {
	startedAt := time.Unix(100, 0).UTC()
	finishedAt := time.Unix(200, 0).UTC()

	w := &mockRunWriter{}

	runs := []SuiteRun{
		{
			PromptName:    "p1",
			PromptVersion: "v1",
			Suite:         &testcase.TestSuite{Suite: "s1"},
			Result: &runner.SuiteResult{
				TotalCases:   1,
				PassedCases:  1,
				FailedCases:  0,
				PassRate:     1,
				AvgScore:     1,
				TotalLatency: 11,
				TotalTokens:  22,
				Results: []runner.RunResult{{
					CaseID:     "c1",
					Passed:     true,
					Score:      1,
					PassAtK:    1,
					PassExpK:   1,
					LatencyMs:  11,
					TokensUsed: 22,
				}},
			},
		},
		{
			PromptName:    "p1",
			PromptVersion: "v1",
			Suite:         &testcase.TestSuite{Suite: "s2"},
			Result: &runner.SuiteResult{
				TotalCases:   1,
				PassedCases:  0,
				FailedCases:  1,
				PassRate:     0,
				AvgScore:     0.2,
				TotalLatency: 33,
				TotalTokens:  44,
				Results: []runner.RunResult{{
					CaseID:     "c2",
					Passed:     false,
					Score:      0.2,
					PassAtK:    0,
					PassExpK:   0,
					LatencyMs:  33,
					TokensUsed: 44,
					Error:      errors.New("boom"),
				}},
			},
		},
	}
	anyFailed, summary := SummarizeRuns(runs)
	if !anyFailed {
		t.Fatalf("SummarizeRuns: expected anyFailed")
	}

	rec, err := SaveRun(nil, w, runs, summary, startedAt, finishedAt, map[string]any{"k": "v"})
	if err != nil {
		t.Fatalf("SaveRun: %v", err)
	}
	if rec == nil {
		t.Fatalf("SaveRun: nil record")
	}
	if !strings.HasPrefix(rec.ID, "run_") {
		t.Fatalf("RunRecord.ID: got %q", rec.ID)
	}
	if rec.TotalSuites != 2 { // uses summary.TotalSuites
		t.Fatalf("TotalSuites: got %d want %d", rec.TotalSuites, 2)
	}
	if rec.PassedSuites != 1 || rec.FailedSuites != 1 {
		t.Fatalf("suite counts: passed=%d failed=%d", rec.PassedSuites, rec.FailedSuites)
	}
	if w.lastCtx == nil {
		t.Fatalf("writer ctx: nil")
	}
	if len(w.runs) != 1 || len(w.suites) != 2 {
		t.Fatalf("writer saved: runs=%d suites=%d", len(w.runs), len(w.suites))
	}
	if got := w.suites[1].CaseResults[0].Error; got != "boom" {
		t.Fatalf("case error: got %q want %q", got, "boom")
	}

	_, err = SaveRun(context.Background(), nil, runs, summary, startedAt, finishedAt, nil)
	if err == nil {
		t.Fatalf("SaveRun(nil writer): expected error")
	}

	w2 := &mockRunWriter{runErr: errors.New("save run")}
	_, err = SaveRun(context.Background(), w2, runs, summary, startedAt, finishedAt, nil)
	if err == nil || !strings.Contains(err.Error(), "save run") {
		t.Fatalf("SaveRun(run err): got %v", err)
	}

	w3 := &mockRunWriter{suiteErrAt: 1}
	_, err = SaveRun(context.Background(), w3, runs, summary, startedAt, finishedAt, nil)
	if err == nil || !strings.Contains(err.Error(), "save suite result") {
		t.Fatalf("SaveRun(suite err): got %v", err)
	}

	w4 := &mockRunWriter{}
	_, err = SaveRun(context.Background(), w4, []SuiteRun{{Suite: &testcase.TestSuite{Suite: "s"}}}, summary, startedAt, finishedAt, nil)
	if err == nil || !strings.Contains(err.Error(), "missing suite result") {
		t.Fatalf("SaveRun(missing result): got %v", err)
	}

	w5 := &mockRunWriter{}
	_, err = SaveRun(context.Background(), w5, []SuiteRun{{Result: &runner.SuiteResult{TotalCases: 1}}}, summary, startedAt, finishedAt, nil)
	if err == nil || !strings.Contains(err.Error(), "missing suite result") {
		t.Fatalf("SaveRun(missing suite): got %v", err)
	}
}

func TestSaveRun_RunIDError(t *testing.T) {
	oldReader := rand.Reader
	rand.Reader = errReader{}
	t.Cleanup(func() { rand.Reader = oldReader })

	_, err := SaveRun(context.Background(), &mockRunWriter{}, nil, RunSummary{}, time.Time{}, time.Time{}, nil)
	if err == nil || !strings.Contains(err.Error(), "run: generate run id") {
		t.Fatalf("SaveRun(run id error): got %v", err)
	}
}

type mockRunWriter struct {
	lastCtx    context.Context
	runs       []*store.RunRecord
	suites     []*store.SuiteRecord
	runErr     error
	suiteErrAt int
	suiteSaves int
}

func (w *mockRunWriter) SaveRun(ctx context.Context, run *store.RunRecord) error {
	w.lastCtx = ctx
	w.runs = append(w.runs, run)
	return w.runErr
}

func (w *mockRunWriter) SaveSuiteResult(ctx context.Context, result *store.SuiteRecord) error {
	w.lastCtx = ctx
	w.suiteSaves++
	if w.suiteErrAt > 0 && w.suiteSaves == w.suiteErrAt {
		return errors.New("save suite")
	}
	w.suites = append(w.suites, result)
	return nil
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) {
	return 0, errors.New("rand fail")
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.TrimSpace(content)+"\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(%s): %v", path, err)
	}
}
