package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestResolveCIMode_Forced(t *testing.T) {
	t.Parallel()

	if !resolveCIMode(&runOptions{ci: true}) {
		t.Fatalf("expected CI mode when --ci is set")
	}
}

func TestApplyCIOutputDefaults(t *testing.T) {
	t.Parallel()

	opts := &runOptions{}
	applyCIOutputDefaults(opts, false)
	if opts.output != "" {
		t.Fatalf("unexpected output change: %q", opts.output)
	}

	applyCIOutputDefaults(opts, true)
	if opts.output != string(FormatGitHub) {
		t.Fatalf("expected github output default, got %q", opts.output)
	}
}

func TestBuildCIReportAndMarkdown(t *testing.T) {
	t.Parallel()

	started := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
	finished := started.Add(2 * time.Second)

	runs := []app.SuiteRun{
		{
			PromptName:    "p1",
			PromptVersion: "v1",
			Suite:         &testcase.TestSuite{Suite: "suite_from_suite"},
			Result:        &runner.SuiteResult{Suite: "suite_from_result", TotalCases: 2, PassedCases: 2, FailedCases: 0, PassRate: 1, AvgScore: 1, TotalLatency: 10, TotalTokens: 20},
		},
		{
			PromptName: "p2",
			Result:     nil,
		},
		{
			PromptName: "p3",
			Suite:      nil,
			Result:     &runner.SuiteResult{Suite: "suite_only_from_result"},
		},
	}

	summary := app.RunSummary{TotalSuites: 3, TotalCases: 3, PassedCases: 2, FailedCases: 1}
	report := buildCIReport(runs, summary, started, finished, 0.8)
	if report.StartedAt == "" || report.FinishedAt == "" {
		t.Fatalf("expected timestamps, got %#v", report)
	}
	if len(report.Suites) != 3 {
		t.Fatalf("expected 3 suites, got %d", len(report.Suites))
	}
	if report.Suites[1].Error == "" {
		t.Fatalf("expected error entry for nil result")
	}

	md := buildCIMarkdown(report)
	if !strings.Contains(md, "Threshold: 0.80") {
		t.Fatalf("expected threshold in markdown, got %q", md)
	}
	if !strings.Contains(md, "| Prompt | Suite |") {
		t.Fatalf("expected table header, got %q", md)
	}
}

func TestBuildCIMarkdown_NoSuites(t *testing.T) {
	t.Parallel()

	md := buildCIMarkdown(ciReport{Threshold: -1, Summary: app.RunSummary{}})
	if !strings.Contains(md, "_No suites run._") {
		t.Fatalf("expected no suites message, got %q", md)
	}
	if strings.Contains(md, "Threshold:") {
		t.Fatalf("did not expect threshold line, got %q", md)
	}
}

func TestWriteCIReportFile(t *testing.T) {
	t.Parallel()

	if err := writeCIReportFile("   ", ciReport{}); err == nil {
		t.Fatalf("expected error for empty path")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "ci.json")
	if err := writeCIReportFile(path, ciReport{StartedAt: "x"}); err != nil {
		t.Fatalf("writeCIReportFile: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected file to exist: %v", err)
	}
}

func TestWriteCIReportFile_MarshalError(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "ci.json")
	if err := writeCIReportFile(path, ciReport{Threshold: math.NaN()}); err == nil {
		t.Fatalf("expected marshal error")
	}
}

func TestEscapeMarkdownCell(t *testing.T) {
	t.Parallel()

	if got := escapeMarkdownCell(" a|\r\nb "); got != "a\\|  b" {
		t.Fatalf("escapeMarkdownCell: got %q", got)
	}
	if got := escapeMarkdownCell("   "); got != "-" {
		t.Fatalf("escapeMarkdownCell(empty): got %q", got)
	}
}

func TestPostPRComment_Errors(t *testing.T) {
	t.Parallel()

	if err := postPRComment(" "); err == nil {
		t.Fatalf("expected error for empty report path")
	}
	if err := postPRComment(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatalf("expected error for missing report file")
	}
}

func TestPostPRComment_Success(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	dir := t.TempDir()
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	if err := os.MkdirAll("scripts", 0o755); err != nil {
		t.Fatalf("MkdirAll(scripts): %v", err)
	}
	if err := os.WriteFile(filepath.Join("scripts", "pr-comment.sh"), []byte("#!/usr/bin/env bash\nexit 0\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(pr-comment.sh): %v", err)
	}

	reportPath := filepath.Join(dir, "report.json")
	if err := os.WriteFile(reportPath, []byte(`{"ok":true}`), 0o644); err != nil {
		t.Fatalf("WriteFile(report.json): %v", err)
	}

	if err := postPRComment(reportPath); err != nil {
		t.Fatalf("postPRComment: %v", err)
	}
}

func TestWriteCIArtifacts_Success(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	dir := t.TempDir()
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	t.Setenv("GITHUB_STEP_SUMMARY", filepath.Join(dir, "summary.md"))

	if err := os.MkdirAll("scripts", 0o755); err != nil {
		t.Fatalf("MkdirAll(scripts): %v", err)
	}
	if err := os.WriteFile(filepath.Join("scripts", "pr-comment.sh"), []byte("#!/usr/bin/env bash\nexit 0\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(pr-comment.sh): %v", err)
	}

	runs := []app.SuiteRun{{
		PromptName:    "p1",
		PromptVersion: "v1",
		Suite:         &testcase.TestSuite{Suite: "s1"},
		Result: &runner.SuiteResult{
			TotalCases:   1,
			PassedCases:  1,
			FailedCases:  0,
			PassRate:     1,
			AvgScore:     1,
			TotalLatency: 1,
			TotalTokens:  1,
		},
	}}
	summary := app.RunSummary{TotalSuites: 1, TotalCases: 1, PassedCases: 1, FailedCases: 0}
	started := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
	finished := started.Add(time.Second)

	writeCIArtifacts(runs, summary, started, finished, 0.5)

	if _, err := os.Stat(ciReportPath); err != nil {
		t.Fatalf("expected report %q to exist: %v", ciReportPath, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "summary.md")); err != nil {
		t.Fatalf("expected job summary to exist: %v", err)
	}
}

func TestWriteCIArtifacts_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	dir := t.TempDir()
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	// Force ci.SetJobSummary() to fail by pointing it at a directory.
	summaryDir := filepath.Join(dir, "summarydir")
	if err := os.MkdirAll(summaryDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(summarydir): %v", err)
	}
	t.Setenv("GITHUB_STEP_SUMMARY", summaryDir)

	started := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
	finished := started.Add(time.Second)

	runs := []app.SuiteRun{{
		PromptName:    "p1",
		PromptVersion: "v1",
		Suite:         &testcase.TestSuite{Suite: "s1"},
		Result: &runner.SuiteResult{
			TotalCases:   1,
			PassedCases:  1,
			FailedCases:  0,
			PassRate:     1,
			AvgScore:     1,
			TotalLatency: 1,
			TotalTokens:  1,
		},
	}}
	summary := app.RunSummary{TotalSuites: 1, TotalCases: 1, PassedCases: 1, FailedCases: 0}

	// Force writeCIReportFile() to fail by blocking "data/" with a file.
	if err := os.WriteFile("data", []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile(data): %v", err)
	}
	writeCIArtifacts(runs, summary, started, finished, 0.5)

	_ = os.Remove("data")
	if err := os.MkdirAll("data", 0o755); err != nil {
		t.Fatalf("MkdirAll(data): %v", err)
	}

	// Let report write succeed but force postPRComment() to fail (missing script).
	writeCIArtifacts(runs, summary, started, finished, 0.5)
	if _, err := os.Stat(ciReportPath); err != nil {
		t.Fatalf("expected report %q to exist: %v", ciReportPath, err)
	}
}
