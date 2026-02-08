package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestPrintRunJSON(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&buf)

	runs := []app.SuiteRun{
		{
			PromptName: "p1",
			Suite:      &testcase.TestSuite{Suite: "s1"},
			Result:     nil,
		},
		{
			PromptName: "p2",
			Suite:      &testcase.TestSuite{Suite: "s2"},
			Result:     &runner.SuiteResult{Suite: "s2", TotalCases: 1, PassedCases: 1, FailedCases: 0, PassRate: 1, AvgScore: 1},
		},
	}
	summary := app.RunSummary{TotalSuites: 2, TotalCases: 1, PassedCases: 1}

	if err := printRunJSON(cmd, runs, summary); err != nil {
		t.Fatalf("printRunJSON: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 json lines, got %d: %q", len(lines), buf.String())
	}

	var first map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("unmarshal first line: %v", err)
	}
	if first["error"] == "" || first["suite"] != "s1" {
		t.Fatalf("expected error+suite in first line, got %#v", first)
	}

	var last map[string]any
	if err := json.Unmarshal([]byte(lines[2]), &last); err != nil {
		t.Fatalf("unmarshal summary line: %v", err)
	}
	if _, ok := last["summary"]; !ok {
		t.Fatalf("expected summary in last line, got %#v", last)
	}
}

func TestSaveRunToStore(t *testing.T) {
	t.Parallel()

	if err := saveRunToStore(context.Background(), nil, nil, app.RunSummary{}, time.Time{}, time.Time{}, nil, false, FormatTable, 1, 0.5, 1); err == nil {
		t.Fatalf("expected error for nil cli state")
	}
	if err := saveRunToStore(context.Background(), &cliState{}, nil, app.RunSummary{}, time.Time{}, time.Time{}, nil, false, FormatTable, 1, 0.5, 1); err == nil {
		t.Fatalf("expected error for nil config")
	}

	st := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "nope"}}}
	if err := saveRunToStore(context.Background(), st, nil, app.RunSummary{}, time.Time{}, time.Time{}, nil, false, FormatTable, 1, 0.5, 1); err == nil || !strings.Contains(err.Error(), "run: open store") {
		t.Fatalf("expected open store error, got %v", err)
	}

	st.cfg.Storage.Type = "memory"
	runs := []app.SuiteRun{{
		PromptName:    "p1",
		PromptVersion: "v1",
		Suite:         &testcase.TestSuite{Suite: "s1"},
		Result:        &runner.SuiteResult{TotalCases: 1, PassedCases: 1, FailedCases: 0, PassRate: 1},
	}}
	summary := app.RunSummary{TotalSuites: 1, TotalCases: 1, PassedCases: 1, FailedCases: 0}
	started := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
	finished := started.Add(time.Second)

	if err := saveRunToStore(nil, st, runs, summary, started, finished, []string{"p1"}, false, FormatJSON, 1, 0.8, 2); err != nil {
		t.Fatalf("saveRunToStore: %v", err)
	}
}
