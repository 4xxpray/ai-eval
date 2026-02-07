package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/ci"
)

const ciReportPath = "data/ci-results.json"

type ciReport struct {
	StartedAt  string          `json:"started_at"`
	FinishedAt string          `json:"finished_at"`
	Threshold  float64         `json:"threshold"`
	Summary    jsonRunSummary  `json:"summary"`
	Suites     []ciSuiteReport `json:"suites"`
}

type ciSuiteReport struct {
	Prompt        string  `json:"prompt"`
	PromptVersion string  `json:"prompt_version,omitempty"`
	Suite         string  `json:"suite"`
	TotalCases    int     `json:"total_cases"`
	PassedCases   int     `json:"passed_cases"`
	FailedCases   int     `json:"failed_cases"`
	PassRate      float64 `json:"pass_rate"`
	AvgScore      float64 `json:"avg_score"`
	TotalLatency  int64   `json:"total_latency_ms"`
	TotalTokens   int     `json:"total_tokens"`
	Passed        bool    `json:"passed"`
	Error         string  `json:"error,omitempty"`
}

func resolveCIMode(opts *runOptions) bool {
	if opts != nil && opts.ci {
		return true
	}
	return ci.DetectCI()
}

func applyCIOutputDefaults(opts *runOptions, ciMode bool) {
	if opts == nil || !ciMode {
		return
	}
	if strings.TrimSpace(opts.output) == "" {
		opts.output = string(FormatGitHub)
	}
}

func writeCIArtifacts(runs []suiteRun, summary runSummary, startedAt, finishedAt time.Time, threshold float64) {
	report := buildCIReport(runs, summary, startedAt, finishedAt, threshold)
	if err := ci.SetJobSummary(buildCIMarkdown(report)); err != nil {
		fmt.Fprintf(os.Stderr, "ci: write job summary: %v\n", err)
	}
	if err := writeCIReportFile(ciReportPath, report); err != nil {
		fmt.Fprintf(os.Stderr, "ci: write report: %v\n", err)
		return
	}
	if err := postPRComment(ciReportPath); err != nil {
		fmt.Fprintf(os.Stderr, "ci: post PR comment: %v\n", err)
	}
}

func buildCIReport(runs []suiteRun, summary runSummary, startedAt, finishedAt time.Time, threshold float64) ciReport {
	report := ciReport{
		StartedAt:  formatTime(startedAt),
		FinishedAt: formatTime(finishedAt),
		Threshold:  threshold,
		Summary: jsonRunSummary{
			TotalSuites:  summary.totalSuites,
			TotalCases:   summary.totalCases,
			PassedCases:  summary.passedCases,
			FailedCases:  summary.failedCases,
			TotalLatency: summary.totalLatency,
			TotalTokens:  summary.totalTokens,
		},
		Suites: make([]ciSuiteReport, 0, len(runs)),
	}

	for _, r := range runs {
		suiteName := ""
		if r.suite != nil {
			suiteName = r.suite.Suite
		}
		if suiteName == "" && r.result != nil {
			suiteName = r.result.Suite
		}

		entry := ciSuiteReport{
			Prompt:        r.promptName,
			PromptVersion: r.promptVersion,
			Suite:         suiteName,
		}

		if r.result == nil {
			entry.Error = "nil suite result"
			report.Suites = append(report.Suites, entry)
			continue
		}

		entry.TotalCases = r.result.TotalCases
		entry.PassedCases = r.result.PassedCases
		entry.FailedCases = r.result.FailedCases
		entry.PassRate = r.result.PassRate
		entry.AvgScore = r.result.AvgScore
		entry.TotalLatency = r.result.TotalLatency
		entry.TotalTokens = r.result.TotalTokens
		entry.Passed = r.result.FailedCases == 0

		report.Suites = append(report.Suites, entry)
	}

	return report
}

func writeCIReportFile(path string, report ciReport) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return fmt.Errorf("ci: empty report path")
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	b, err := json.Marshal(report)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func buildCIMarkdown(report ciReport) string {
	var buf strings.Builder
	buf.WriteString("## Prompt Evaluation Results\n\n")
	if report.Threshold >= 0 {
		fmt.Fprintf(&buf, "Threshold: %.2f\n\n", report.Threshold)
	}
	fmt.Fprintf(&buf, "Suites: %d | Cases: %d | Passed: %d | Failed: %d\n\n",
		report.Summary.TotalSuites,
		report.Summary.TotalCases,
		report.Summary.PassedCases,
		report.Summary.FailedCases,
	)

	if len(report.Suites) == 0 {
		buf.WriteString("_No suites run._\n")
		return buf.String()
	}

	buf.WriteString("| Prompt | Suite | Cases | Passed | Failed | Pass Rate | Avg Score | Error |\n")
	buf.WriteString("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
	for _, s := range report.Suites {
		prompt := escapeMarkdownCell(s.Prompt)
		suite := escapeMarkdownCell(s.Suite)
		errMsg := escapeMarkdownCell(s.Error)
		if s.Error != "" {
			fmt.Fprintf(&buf, "| %s | %s | - | - | - | - | - | %s |\n", prompt, suite, errMsg)
			continue
		}
		fmt.Fprintf(&buf, "| %s | %s | %d | %d | %d | %.3f | %.3f | - |\n",
			prompt,
			suite,
			s.TotalCases,
			s.PassedCases,
			s.FailedCases,
			s.PassRate,
			s.AvgScore,
		)
	}

	return buf.String()
}

func escapeMarkdownCell(s string) string {
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.TrimSpace(s)
	if s == "" {
		return "-"
	}
	return s
}

func postPRComment(reportPath string) error {
	reportPath = strings.TrimSpace(reportPath)
	if reportPath == "" {
		return fmt.Errorf("ci: missing report path")
	}
	if _, err := os.Stat(reportPath); err != nil {
		return err
	}
	scriptPath := filepath.Join("scripts", "pr-comment.sh")
	if _, err := os.Stat(scriptPath); err != nil {
		return err
	}

	cmd := exec.Command("bash", scriptPath, reportPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
