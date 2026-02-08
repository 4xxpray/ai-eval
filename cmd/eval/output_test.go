package main

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/runner"
)

func TestParseOutputFormat(t *testing.T) {
	t.Parallel()

	tests := []struct {
		in   string
		want OutputFormat
	}{
		{in: "table", want: FormatTable},
		{in: " TABLE ", want: FormatTable},
		{in: "json", want: FormatJSON},
		{in: "jsonl", want: FormatJSON},
		{in: "github", want: FormatGitHub},
		{in: "gh", want: FormatGitHub},
		{in: "nope", want: ""},
	}

	for _, tt := range tests {
		if got := parseOutputFormat(tt.in); got != tt.want {
			t.Fatalf("parseOutputFormat(%q): got %q want %q", tt.in, got, tt.want)
		}
	}
}

func TestResolveOutputFormat(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		flagValue   string
		configValue string
		all         bool
		want        OutputFormat
		wantErrSub  string
	}{
		{name: "flag invalid", flagValue: "wat", wantErrSub: "invalid --output"},
		{name: "flag table with all", flagValue: "table", all: true, wantErrSub: "--all does not support"},
		{name: "flag json with all", flagValue: "json", all: true, want: FormatJSON},
		{name: "config table with all => json", configValue: "table", all: true, want: FormatJSON},
		{name: "config github with all", configValue: "github", all: true, want: FormatGitHub},
		{name: "config invalid with all => json", configValue: "wat", all: true, want: FormatJSON},
		{name: "config invalid without all => table", configValue: "wat", want: FormatTable},
		{name: "default table", want: FormatTable},
		{name: "default json when all", all: true, want: FormatJSON},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := resolveOutputFormat(tt.flagValue, tt.configValue, tt.all)
			if tt.wantErrSub != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrSub) {
					t.Fatalf("resolveOutputFormat: err=%v want substring %q", err, tt.wantErrSub)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveOutputFormat: %v", err)
			}
			if got != tt.want {
				t.Fatalf("resolveOutputFormat: got %q want %q", got, tt.want)
			}
		})
	}
}

func TestColoredStatus(t *testing.T) {
	t.Parallel()

	if got := coloredStatus(true); !strings.Contains(got, "PASS") {
		t.Fatalf("coloredStatus(true): got %q", got)
	}
	if got := coloredStatus(false); !strings.Contains(got, "FAIL") {
		t.Fatalf("coloredStatus(false): got %q", got)
	}
}

func TestFormatSuiteResult(t *testing.T) {
	t.Parallel()

	if got := FormatSuiteResult(nil, FormatTable); !strings.Contains(got, "Suite: <nil>") {
		t.Fatalf("FormatSuiteResult(nil, table): got %q", got)
	}
	if got := FormatSuiteResult(sampleSuiteResult(), FormatTable); !strings.Contains(got, "boom") {
		t.Fatalf("FormatSuiteResult(table): expected error text, got %q", got)
	}
	if got := FormatSuiteResult(sampleSuiteResult(), OutputFormat("wat")); !strings.Contains(got, "unknown output format") {
		t.Fatalf("FormatSuiteResult(unknown): got %q", got)
	}
	if got := FormatCompareResult(nil, nil, OutputFormat("wat")); !strings.Contains(got, "unknown output format") {
		t.Fatalf("FormatCompareResult(unknown): got %q", got)
	}
}

func TestFormatSuiteJSONAndGitHub(t *testing.T) {
	t.Parallel()

	res := sampleSuiteResult()

	gotJSON := formatSuiteJSON(res)
	var parsed jsonSuiteResult
	if err := json.Unmarshal([]byte(strings.TrimSpace(gotJSON)), &parsed); err != nil {
		t.Fatalf("formatSuiteJSON: unmarshal: %v", err)
	}
	if parsed.Suite != res.Suite {
		t.Fatalf("formatSuiteJSON.suite: got %q want %q", parsed.Suite, res.Suite)
	}
	if parsed.FailedCases != 1 || parsed.PassedCases != 1 || parsed.TotalCases != 2 {
		t.Fatalf("formatSuiteJSON counts: got %#v", parsed)
	}

	if got := formatSuiteJSON(nil); !strings.Contains(got, "nil suite result") {
		t.Fatalf("formatSuiteJSON(nil): got %q", got)
	}

	if got := formatSuiteJSON(&runner.SuiteResult{Suite: "s", PassRate: math.NaN()}); !strings.Contains(got, "\"error\"") {
		t.Fatalf("formatSuiteJSON(NaN): got %q", got)
	}

	gotGH := formatSuiteGitHub(res)
	if !strings.Contains(gotGH, "::error::") {
		t.Fatalf("formatSuiteGitHub: expected annotation, got %q", gotGH)
	}
	if !strings.Contains(gotGH, "Summary: suite=") {
		t.Fatalf("formatSuiteGitHub: expected summary, got %q", gotGH)
	}

	if got := formatSuiteGitHub(nil); !strings.Contains(got, "nil suite result") {
		t.Fatalf("formatSuiteGitHub(nil): got %q", got)
	}
}

func TestSanitizeGitHubAnnotation(t *testing.T) {
	t.Parallel()

	got := sanitizeGitHubAnnotation(" a\r\nb \n")
	if got != "a  b" {
		t.Fatalf("sanitizeGitHubAnnotation: got %q want %q", got, "a  b")
	}
}

func TestBuildCompareAndFormats(t *testing.T) {
	t.Parallel()

	summary, diffs := buildCompare(nil, sampleSuiteResult())
	if summary.Suite != "<nil>" || !summary.Regressed || len(diffs) != 0 {
		t.Fatalf("buildCompare(nil): got %#v diffs=%v", summary, diffs)
	}

	v1 := &runner.SuiteResult{
		Suite:    " ",
		PassRate: 1,
		AvgScore: 1,
		Results: []runner.RunResult{
			{CaseID: "c1", Passed: true, Score: 1, Error: errors.New("v1")},
			{CaseID: "", Passed: true, Score: 1},
			{CaseID: "dup", Passed: true, Score: 1},
			{CaseID: "dup", Passed: false, Score: 0.2},
			{CaseID: "only_v1", Passed: true, Score: 1},
		},
	}
	v2 := &runner.SuiteResult{
		Suite:    "s2",
		PassRate: 0,
		AvgScore: 0,
		Results: []runner.RunResult{
			{CaseID: "", Passed: true, Score: 1},
			{CaseID: "c1", Passed: false, Score: 0.1, Error: errors.New("v2")},
			{CaseID: "only_v2", Passed: true, Score: 1},
		},
	}

	summary, diffs = buildCompare(v1, v2)
	if summary.Suite != "s2" {
		t.Fatalf("suite name fallback: got %q want %q", summary.Suite, "s2")
	}
	if summary.MissingInV1 != 1 || summary.MissingInV2 != 2 || len(summary.MissingCaseIDs) != 3 {
		t.Fatalf("missing cases: %#v", summary)
	}
	if !summary.Regressed || summary.RegressionCnt != 1 {
		t.Fatalf("regression summary: %#v", summary)
	}
	if len(diffs) != 1 || diffs[0].CaseID != "c1" || !diffs[0].Regression {
		t.Fatalf("diffs: %#v", diffs)
	}
	if diffs[0].V1Error == "" || diffs[0].V2Error == "" {
		t.Fatalf("errors not captured: %#v", diffs[0])
	}

	table := formatCompareTable(v1, v2)
	if !strings.Contains(table, "Missing cases:") || !strings.Contains(table, "Regression:") {
		t.Fatalf("formatCompareTable: got %q", table)
	}
	gh := formatCompareGitHub(v1, v2)
	if !strings.Contains(gh, "::warning::") {
		t.Fatalf("formatCompareGitHub: got %q", gh)
	}
}

func TestIsRegression_ScoreDelta(t *testing.T) {
	t.Parallel()

	if !isRegression(compareCaseDiff{V1Passed: false, V2Passed: false, ScoreDelta: -0.1}) {
		t.Fatalf("expected negative score delta to be regression")
	}
}

func TestFormatCompareJSON_MarshalError(t *testing.T) {
	t.Parallel()

	v1 := &runner.SuiteResult{Suite: "s", PassRate: math.NaN()}
	v2 := &runner.SuiteResult{Suite: "s", PassRate: 0}
	if got := formatCompareJSON(v1, v2); !strings.Contains(got, "\"error\"") {
		t.Fatalf("formatCompareJSON(NaN): got %q", got)
	}
}

func sampleSuiteResult() *runner.SuiteResult {
	return &runner.SuiteResult{
		Suite:        "suite",
		TotalCases:   2,
		PassedCases:  1,
		FailedCases:  1,
		PassRate:     0.5,
		AvgScore:     0.75,
		TotalLatency: 30,
		TotalTokens:  300,
		Results: []runner.RunResult{
			{
				Suite:      "suite",
				CaseID:     "c1",
				Passed:     true,
				Score:      1,
				PassAtK:    1,
				PassExpK:   1,
				LatencyMs:  10,
				TokensUsed: 100,
				Trials: []runner.TrialResult{{
					TrialNum:  1,
					Response:  "ok",
					ToolCalls: []llm.ToolUse{{ID: "t1", Name: "search", Input: map[string]any{"q": "x"}}},
					Evaluations: []evaluator.Result{{
						Passed:  true,
						Score:   1,
						Message: "ok",
					}},
					Passed:    true,
					Score:     1,
					LatencyMs: 10,
				}},
			},
			{
				Suite:      "suite",
				CaseID:     "c2",
				Passed:     false,
				Score:      0.5,
				PassAtK:    0,
				PassExpK:   0,
				LatencyMs:  20,
				TokensUsed: 200,
				Error:      errors.New("boom"),
			},
		},
	}
}
