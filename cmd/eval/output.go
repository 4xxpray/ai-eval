package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/runner"
)

type OutputFormat string

const (
	FormatTable  OutputFormat = "table"
	FormatJSON   OutputFormat = "json"
	FormatGitHub OutputFormat = "github"
)

const (
	colorReset = "\033[0m"
	colorRed   = "\033[31m"
	colorGreen = "\033[32m"
)

func parseOutputFormat(s string) OutputFormat {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "table":
		return FormatTable
	case "json", "jsonl":
		return FormatJSON
	case "github", "gh":
		return FormatGitHub
	default:
		return ""
	}
}

func resolveOutputFormat(flagValue string, configValue string, all bool) (OutputFormat, error) {
	if strings.TrimSpace(flagValue) != "" {
		out := parseOutputFormat(flagValue)
		if out == "" {
			return "", fmt.Errorf("invalid --output %q (expected table|json|github)", flagValue)
		}
		if all && out == FormatTable {
			return "", fmt.Errorf("--all does not support --output table")
		}
		return out, nil
	}

	if out := parseOutputFormat(configValue); out != "" {
		if all && out == FormatTable {
			return FormatJSON, nil
		}
		return out, nil
	}

	if all {
		return FormatJSON, nil
	}
	return FormatTable, nil
}

func coloredStatus(passed bool) string {
	if passed {
		return colorGreen + "PASS" + colorReset
	}
	return colorRed + "FAIL" + colorReset
}

func suitePassed(res *runner.SuiteResult) bool {
	return res != nil && res.FailedCases == 0
}

func FormatSuiteResult(result *runner.SuiteResult, format OutputFormat) string {
	switch format {
	case FormatTable:
		return formatSuiteTable(result)
	case FormatJSON:
		return formatSuiteJSON(result)
	case FormatGitHub:
		return formatSuiteGitHub(result)
	default:
		return fmt.Sprintf("error: unknown output format %q\n", format)
	}
}

func FormatCompareResult(v1, v2 *runner.SuiteResult, format OutputFormat) string {
	switch format {
	case FormatTable:
		return formatCompareTable(v1, v2)
	case FormatJSON:
		return formatCompareJSON(v1, v2)
	case FormatGitHub:
		return formatCompareGitHub(v1, v2)
	default:
		return fmt.Sprintf("error: unknown output format %q\n", format)
	}
}

func formatSuiteTable(result *runner.SuiteResult) string {
	if result == nil {
		return "Suite: <nil> " + coloredStatus(false) + "\n\n"
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Suite: %s %s\n", result.Suite, coloredStatus(suitePassed(result)))
	fmt.Fprintf(&buf, "Cases: %d passed=%d failed=%d pass_rate=%.2f avg_score=%.2f latency_ms=%d tokens=%d\n",
		result.TotalCases, result.PassedCases, result.FailedCases, result.PassRate, result.AvgScore, result.TotalLatency, result.TotalTokens)

	tw := tabwriter.NewWriter(&buf, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "CASE\tRESULT\tSCORE\tPASS@K\tLAT(ms)\tTOKENS\tERROR")
	for _, rr := range result.Results {
		errMsg := ""
		if rr.Error != nil {
			errMsg = rr.Error.Error()
		}
		fmt.Fprintf(tw, "%s\t%s\t%.3f\t%.3f\t%d\t%d\t%s\n",
			rr.CaseID, coloredStatus(rr.Passed), rr.Score, rr.PassAtK, rr.LatencyMs, rr.TokensUsed, errMsg)
	}
	_ = tw.Flush()
	buf.WriteByte('\n')
	return buf.String()
}

type jsonSuiteResult struct {
	Suite        string        `json:"suite"`
	Passed       bool          `json:"passed"`
	TotalCases   int           `json:"total_cases"`
	PassedCases  int           `json:"passed_cases"`
	FailedCases  int           `json:"failed_cases"`
	PassRate     float64       `json:"pass_rate"`
	AvgScore     float64       `json:"avg_score"`
	TotalLatency int64         `json:"total_latency_ms"`
	TotalTokens  int           `json:"total_tokens"`
	Cases        []jsonCaseRun `json:"cases"`
}

type jsonCaseRun struct {
	CaseID     string         `json:"case_id"`
	Passed     bool           `json:"passed"`
	Score      float64        `json:"score"`
	PassAtK    float64        `json:"pass_at_k"`
	PassExpK   float64        `json:"pass_exp_k"`
	LatencyMs  int64          `json:"latency_ms"`
	TokensUsed int            `json:"tokens_used"`
	Error      string         `json:"error,omitempty"`
	Trials     []jsonTrialRun `json:"trials,omitempty"`
}

type jsonTrialRun struct {
	TrialNum    int                `json:"trial_num"`
	Response    string             `json:"response"`
	ToolCalls   []llm.ToolUse      `json:"tool_calls,omitempty"`
	Evaluations []evaluator.Result `json:"evaluations,omitempty"`
	Passed      bool               `json:"passed"`
	Score       float64            `json:"score"`
	LatencyMs   int64              `json:"latency_ms"`
}

func suiteResultToJSON(result *runner.SuiteResult) jsonSuiteResult {
	out := jsonSuiteResult{
		Suite:        result.Suite,
		Passed:       suitePassed(result),
		TotalCases:   result.TotalCases,
		PassedCases:  result.PassedCases,
		FailedCases:  result.FailedCases,
		PassRate:     result.PassRate,
		AvgScore:     result.AvgScore,
		TotalLatency: result.TotalLatency,
		TotalTokens:  result.TotalTokens,
		Cases:        make([]jsonCaseRun, 0, len(result.Results)),
	}

	for _, rr := range result.Results {
		caseOut := jsonCaseRun{
			CaseID:     rr.CaseID,
			Passed:     rr.Passed,
			Score:      rr.Score,
			PassAtK:    rr.PassAtK,
			PassExpK:   rr.PassExpK,
			LatencyMs:  rr.LatencyMs,
			TokensUsed: rr.TokensUsed,
			Trials:     make([]jsonTrialRun, 0, len(rr.Trials)),
		}
		if rr.Error != nil {
			caseOut.Error = rr.Error.Error()
		}
		for _, tr := range rr.Trials {
			caseOut.Trials = append(caseOut.Trials, jsonTrialRun{
				TrialNum:    tr.TrialNum,
				Response:    tr.Response,
				ToolCalls:   tr.ToolCalls,
				Evaluations: tr.Evaluations,
				Passed:      tr.Passed,
				Score:       tr.Score,
				LatencyMs:   tr.LatencyMs,
			})
		}
		out.Cases = append(out.Cases, caseOut)
	}

	return out
}

func formatSuiteJSON(result *runner.SuiteResult) string {
	if result == nil {
		return "{\"error\":\"nil suite result\"}\n"
	}

	out := suiteResultToJSON(result)

	b, err := json.Marshal(out)
	if err != nil {
		return fmt.Sprintf("{\"error\":%q}\n", err.Error())
	}
	return string(b) + "\n"
}

func formatSuiteGitHub(result *runner.SuiteResult) string {
	if result == nil {
		return "::error::nil suite result\n"
	}

	var buf strings.Builder
	for _, rr := range result.Results {
		if rr.Passed {
			continue
		}
		msg := fmt.Sprintf("suite=%s case=%s score=%.3f pass@k=%.3f", result.Suite, rr.CaseID, rr.Score, rr.PassAtK)
		if rr.Error != nil {
			msg += " error=" + rr.Error.Error()
		}
		buf.WriteString("::error::")
		buf.WriteString(sanitizeGitHubAnnotation(msg))
		buf.WriteByte('\n')
	}

	buf.WriteString(fmt.Sprintf("Summary: suite=%s cases=%d passed=%d failed=%d pass_rate=%.3f avg_score=%.3f\n",
		result.Suite, result.TotalCases, result.PassedCases, result.FailedCases, result.PassRate, result.AvgScore))
	return buf.String()
}

func sanitizeGitHubAnnotation(s string) string {
	// GitHub Actions commands treat CR/LF and percent-encoding specially.
	// Keep it simple: flatten newlines and carriage returns.
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}

type compareCaseDiff struct {
	CaseID     string
	V1Passed   bool
	V2Passed   bool
	V1Score    float64
	V2Score    float64
	ScoreDelta float64
	V1Error    string
	V2Error    string
	Regression bool
}

type compareSummary struct {
	Suite          string
	V1PassRate     float64
	V2PassRate     float64
	PassRateDelta  float64
	V1AvgScore     float64
	V2AvgScore     float64
	AvgScoreDelta  float64
	Regressed      bool
	RegressionCnt  int
	ComparedCases  int
	MissingInV1    int
	MissingInV2    int
	MissingCaseIDs []string
}

func buildCompare(v1, v2 *runner.SuiteResult) (compareSummary, []compareCaseDiff) {
	var summary compareSummary
	if v1 == nil || v2 == nil {
		summary.Suite = "<nil>"
		summary.Regressed = true
		return summary, nil
	}

	summary.Suite = v1.Suite
	if strings.TrimSpace(summary.Suite) == "" {
		summary.Suite = v2.Suite
	}

	summary.V1PassRate = v1.PassRate
	summary.V2PassRate = v2.PassRate
	summary.PassRateDelta = v2.PassRate - v1.PassRate
	summary.V1AvgScore = v1.AvgScore
	summary.V2AvgScore = v2.AvgScore
	summary.AvgScoreDelta = v2.AvgScore - v1.AvgScore

	v1ByID := make(map[string]runner.RunResult, len(v1.Results))
	for _, rr := range v1.Results {
		v1ByID[rr.CaseID] = rr
	}
	v2ByID := make(map[string]runner.RunResult, len(v2.Results))
	for _, rr := range v2.Results {
		v2ByID[rr.CaseID] = rr
	}

	caseIDs := make([]string, 0, len(v1ByID)+len(v2ByID))
	seen := make(map[string]struct{}, len(v1ByID)+len(v2ByID))
	for _, rr := range v1.Results {
		if rr.CaseID == "" {
			continue
		}
		if _, ok := seen[rr.CaseID]; ok {
			continue
		}
		seen[rr.CaseID] = struct{}{}
		caseIDs = append(caseIDs, rr.CaseID)
	}
	for _, rr := range v2.Results {
		if rr.CaseID == "" {
			continue
		}
		if _, ok := seen[rr.CaseID]; ok {
			continue
		}
		seen[rr.CaseID] = struct{}{}
		caseIDs = append(caseIDs, rr.CaseID)
	}

	diffs := make([]compareCaseDiff, 0, len(caseIDs))
	for _, id := range caseIDs {
		rr1, ok1 := v1ByID[id]
		rr2, ok2 := v2ByID[id]
		if !ok1 {
			summary.MissingInV1++
		}
		if !ok2 {
			summary.MissingInV2++
		}
		if !ok1 || !ok2 {
			summary.MissingCaseIDs = append(summary.MissingCaseIDs, id)
			continue
		}

		d := compareCaseDiff{
			CaseID:     id,
			V1Passed:   rr1.Passed,
			V2Passed:   rr2.Passed,
			V1Score:    rr1.Score,
			V2Score:    rr2.Score,
			ScoreDelta: rr2.Score - rr1.Score,
		}
		if rr1.Error != nil {
			d.V1Error = rr1.Error.Error()
		}
		if rr2.Error != nil {
			d.V2Error = rr2.Error.Error()
		}
		d.Regression = isRegression(d)
		if d.Regression {
			summary.Regressed = true
			summary.RegressionCnt++
		}
		diffs = append(diffs, d)
	}

	summary.ComparedCases = len(diffs)
	return summary, diffs
}

func isRegression(d compareCaseDiff) bool {
	if d.V1Passed && !d.V2Passed {
		return true
	}
	if d.ScoreDelta < -1e-6 {
		return true
	}
	return false
}

func formatCompareTable(v1, v2 *runner.SuiteResult) string {
	summary, diffs := buildCompare(v1, v2)

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Suite: %s\n", summary.Suite)
	fmt.Fprintf(&buf, "PassRate: v1=%.3f v2=%.3f diff=%+.3f\n", summary.V1PassRate, summary.V2PassRate, summary.PassRateDelta)
	fmt.Fprintf(&buf, "AvgScore: v1=%.3f v2=%.3f diff=%+.3f\n", summary.V1AvgScore, summary.V2AvgScore, summary.AvgScoreDelta)

	if summary.MissingInV1 > 0 || summary.MissingInV2 > 0 {
		sort.Strings(summary.MissingCaseIDs)
		fmt.Fprintf(&buf, "Missing cases: only_in_v1=%d only_in_v2=%d ids=%s\n",
			summary.MissingInV2, summary.MissingInV1, strings.Join(summary.MissingCaseIDs, ","))
	}

	tw := tabwriter.NewWriter(&buf, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "CASE\tV1\tV2\tSCORE1\tSCORE2\tΔSCORE\tREGRESSION")
	for _, d := range diffs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%.3f\t%.3f\t%+.3f\t%v\n",
			d.CaseID, passLabel(d.V1Passed), passLabel(d.V2Passed), d.V1Score, d.V2Score, d.ScoreDelta, d.Regression)
	}
	_ = tw.Flush()
	buf.WriteByte('\n')

	if summary.Regressed {
		fmt.Fprintf(&buf, "Regression: %s (cases=%d)\n\n", coloredStatus(false), summary.RegressionCnt)
	} else {
		fmt.Fprintf(&buf, "Regression: %s\n\n", coloredStatus(true))
	}

	return buf.String()
}

func passLabel(passed bool) string {
	if passed {
		return "PASS"
	}
	return "FAIL"
}

type jsonCompareResult struct {
	Suite     string             `json:"suite"`
	V1        jsonCompareSummary `json:"v1"`
	V2        jsonCompareSummary `json:"v2"`
	Diff      jsonCompareDiff    `json:"diff"`
	Cases     []jsonCompareCase  `json:"cases"`
	Regressed bool               `json:"regressed"`
	Meta      jsonCompareMeta    `json:"meta,omitempty"`
}

type jsonCompareSummary struct {
	PassRate float64 `json:"pass_rate"`
	AvgScore float64 `json:"avg_score"`
}

type jsonCompareDiff struct {
	PassRate float64 `json:"pass_rate"`
	AvgScore float64 `json:"avg_score"`
}

type jsonCompareCase struct {
	CaseID     string  `json:"case_id"`
	V1Passed   bool    `json:"v1_passed"`
	V2Passed   bool    `json:"v2_passed"`
	V1Score    float64 `json:"v1_score"`
	V2Score    float64 `json:"v2_score"`
	ScoreDelta float64 `json:"score_delta"`
	Regression bool    `json:"regression"`
	V1Error    string  `json:"v1_error,omitempty"`
	V2Error    string  `json:"v2_error,omitempty"`
}

type jsonCompareMeta struct {
	RegressionCount int      `json:"regression_count"`
	ComparedCases   int      `json:"compared_cases"`
	MissingInV1     int      `json:"missing_in_v1"`
	MissingInV2     int      `json:"missing_in_v2"`
	MissingCaseIDs  []string `json:"missing_case_ids,omitempty"`
}

func formatCompareJSON(v1, v2 *runner.SuiteResult) string {
	summary, diffs := buildCompare(v1, v2)

	out := jsonCompareResult{
		Suite: summary.Suite,
		V1: jsonCompareSummary{
			PassRate: summary.V1PassRate,
			AvgScore: summary.V1AvgScore,
		},
		V2: jsonCompareSummary{
			PassRate: summary.V2PassRate,
			AvgScore: summary.V2AvgScore,
		},
		Diff: jsonCompareDiff{
			PassRate: summary.PassRateDelta,
			AvgScore: summary.AvgScoreDelta,
		},
		Cases:     make([]jsonCompareCase, 0, len(diffs)),
		Regressed: summary.Regressed,
		Meta: jsonCompareMeta{
			RegressionCount: summary.RegressionCnt,
			ComparedCases:   summary.ComparedCases,
			MissingInV1:     summary.MissingInV1,
			MissingInV2:     summary.MissingInV2,
			MissingCaseIDs:  summary.MissingCaseIDs,
		},
	}

	for _, d := range diffs {
		out.Cases = append(out.Cases, jsonCompareCase{
			CaseID:     d.CaseID,
			V1Passed:   d.V1Passed,
			V2Passed:   d.V2Passed,
			V1Score:    d.V1Score,
			V2Score:    d.V2Score,
			ScoreDelta: d.ScoreDelta,
			Regression: d.Regression,
			V1Error:    d.V1Error,
			V2Error:    d.V2Error,
		})
	}

	b, err := json.Marshal(out)
	if err != nil {
		return fmt.Sprintf("{\"error\":%q}\n", err.Error())
	}
	return string(b) + "\n"
}

func formatCompareGitHub(v1, v2 *runner.SuiteResult) string {
	summary, diffs := buildCompare(v1, v2)

	var buf strings.Builder
	for _, d := range diffs {
		if !d.Regression {
			continue
		}
		msg := fmt.Sprintf("regression suite=%s case=%s v1_pass=%v v2_pass=%v score_delta=%+.3f",
			summary.Suite, d.CaseID, d.V1Passed, d.V2Passed, d.ScoreDelta)
		buf.WriteString("::warning::")
		buf.WriteString(sanitizeGitHubAnnotation(msg))
		buf.WriteByte('\n')
	}

	buf.WriteString(fmt.Sprintf("Summary: suite=%s pass_rate_diff=%+.3f avg_score_diff=%+.3f regressions=%d\n",
		summary.Suite, summary.PassRateDelta, summary.AvgScoreDelta, summary.RegressionCnt))
	return buf.String()
}
