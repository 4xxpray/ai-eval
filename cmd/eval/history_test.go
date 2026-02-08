package main

import (
	"bytes"
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestParseSince(t *testing.T) {
	t.Parallel()

	if ts, err := parseSince(" "); err != nil || !ts.IsZero() {
		t.Fatalf("parseSince(empty): ts=%v err=%v", ts, err)
	}

	got, err := parseSince("2026-02-07")
	if err != nil {
		t.Fatalf("parseSince(YYYY-MM-DD): %v", err)
	}
	if got.Format("2006-01-02") != "2026-02-07" {
		t.Fatalf("parseSince(YYYY-MM-DD): got %v", got)
	}

	got, err = parseSince("2026-02-07T00:00:00Z")
	if err != nil {
		t.Fatalf("parseSince(RFC3339): %v", err)
	}
	if got.UTC().Format(time.RFC3339) != "2026-02-07T00:00:00Z" {
		t.Fatalf("parseSince(RFC3339): got %v", got)
	}

	if _, err := parseSince("nope"); err == nil {
		t.Fatalf("expected error for invalid since")
	}
}

func TestFormatTime(t *testing.T) {
	t.Parallel()

	if got := formatTime(time.Time{}); got != "-" {
		t.Fatalf("formatTime(zero): got %q", got)
	}

	ts := time.Date(2026, 2, 7, 1, 2, 3, 0, time.FixedZone("x", 3600))
	if got := formatTime(ts); got != "2026-02-07T00:02:03Z" {
		t.Fatalf("formatTime(non-zero): got %q", got)
	}
}

func TestHistoryCommands(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "ai-eval.sqlite")

	stor, err := store.NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}

	started := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
	finished := started.Add(time.Second)

	runs := []app.SuiteRun{{
		PromptName:    "p1",
		PromptVersion: "v1",
		Suite:         &testcase.TestSuite{Suite: "suite1"},
		Result: &runner.SuiteResult{
			Suite:        "suite1",
			TotalCases:   2,
			PassedCases:  1,
			FailedCases:  1,
			PassRate:     0.5,
			AvgScore:     0.75,
			TotalLatency: 30,
			TotalTokens:  100,
			Results: []runner.RunResult{
				{CaseID: "c1", Passed: true, Score: 1, PassAtK: 1, LatencyMs: 10, TokensUsed: 50},
				{CaseID: "c2", Passed: false, Score: 0.5, PassAtK: 0, LatencyMs: 20, TokensUsed: 50, Error: errors.New("boom")},
			},
		},
	}}
	_, summary := app.SummarizeRuns(runs)
	rec, err := app.SaveRun(context.Background(), stor, runs, summary, started, finished, map[string]any{"x": "y"})
	if err != nil {
		_ = stor.Close()
		t.Fatalf("SaveRun: %v", err)
	}
	_ = stor.Close()

	st := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "sqlite", Path: dbPath}}}

	t.Run("list", func(t *testing.T) {
		var buf bytes.Buffer
		cmd := &cobra.Command{}
		cmd.SetOut(&buf)
		cmd.SetContext(context.Background())

		if err := runHistoryList(cmd, st, &historyOptions{limit: 20}); err != nil {
			t.Fatalf("runHistoryList: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "RUN_ID") || !strings.Contains(out, rec.ID) {
			t.Fatalf("unexpected list output: %q", out)
		}
	})

	t.Run("show", func(t *testing.T) {
		var buf bytes.Buffer
		cmd := &cobra.Command{}
		cmd.SetOut(&buf)
		cmd.SetContext(context.Background())

		if err := runHistoryShow(cmd, st, rec.ID); err != nil {
			t.Fatalf("runHistoryShow: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "Run: "+rec.ID) {
			t.Fatalf("expected run header, got %q", out)
		}
		if !strings.Contains(out, "Suite: suite1") || !strings.Contains(out, "CASE") {
			t.Fatalf("expected suite table, got %q", out)
		}
		if !strings.Contains(out, "c1") || !strings.Contains(out, "c2") {
			t.Fatalf("expected case rows, got %q", out)
		}
	})

	t.Run("show missing", func(t *testing.T) {
		cmd := &cobra.Command{}
		cmd.SetContext(context.Background())

		if err := runHistoryShow(cmd, st, "missing"); err == nil || !strings.Contains(err.Error(), "not found") {
			t.Fatalf("expected not found error, got %v", err)
		}
	})
}

func TestRunHistoryList_NoRuns(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "empty.sqlite")
	st := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "sqlite", Path: dbPath}}}

	var buf bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&buf)
	cmd.SetContext(context.Background())

	if err := runHistoryList(cmd, st, &historyOptions{limit: 1}); err != nil {
		t.Fatalf("runHistoryList(empty): %v", err)
	}
	if !strings.Contains(buf.String(), "No runs found.") {
		t.Fatalf("expected empty message, got %q", buf.String())
	}
}
