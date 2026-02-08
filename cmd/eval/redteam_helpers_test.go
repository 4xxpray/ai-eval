package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/redteam"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestParseRedteamCategories(t *testing.T) {
	t.Parallel()

	got := parseRedteamCategories("")
	if got != nil {
		t.Fatalf("empty: got=%v", got)
	}

	got = parseRedteamCategories("all")
	if len(got) != 4 {
		t.Fatalf("all: got=%v", got)
	}

	got = parseRedteamCategories(" jailbreak, ,pii ")
	if len(got) != 2 || got[0] != redteam.Category("jailbreak") || got[1] != redteam.Category("pii") {
		t.Fatalf("list: got=%v", got)
	}
}

func TestLimitRedteamAttacks(t *testing.T) {
	t.Parallel()

	in := []testcase.TestCase{
		{ID: "c1", Input: map[string]any{"category": "jailbreak"}},
		{ID: "c2", Input: map[string]any{"category": "jailbreak"}},
		{ID: "c3", Input: map[string]any{"category": "pii"}},
		{ID: "c4", Input: map[string]any{"category": ""}},
	}

	if got := limitRedteamAttacks(in, nil, 0); len(got) != len(in) {
		t.Fatalf("perCategory<=0 should not filter: got=%v", got)
	}

	got := limitRedteamAttacks(in, []redteam.Category{redteam.CategoryJailbreak}, 1)
	if len(got) != 1 || got[0].ID != "c1" {
		t.Fatalf("limit: got=%v", got)
	}
}

func TestRenderRedteamSystemPrompt(t *testing.T) {
	t.Parallel()

	if _, err := renderRedteamSystemPrompt(nil); err == nil {
		t.Fatalf("expected error for nil prompt")
	}

	p := &prompt.Prompt{
		Name:     "p",
		Template: "Lang {{.LANG}} Diff {{.DIFF}} Email {{.EMAIL}}",
		Variables: []prompt.Variable{
			{Name: "LANG", Required: true},
			{Name: "DIFF", Required: true},
			{Name: "EMAIL", Required: true},
			{Name: "IGNORED", Required: false, Default: "x"},
			{Name: "", Required: true},
		},
	}

	out, err := renderRedteamSystemPrompt(p)
	if err != nil {
		t.Fatalf("renderRedteamSystemPrompt: %v", err)
	}
	if !strings.Contains(out, "Lang go") {
		t.Fatalf("expected LANG placeholder, got %q", out)
	}
	if !strings.Contains(out, "example@example.com") {
		t.Fatalf("expected EMAIL placeholder, got %q", out)
	}
	if !strings.Contains(out, "diff --git") {
		t.Fatalf("expected DIFF placeholder, got %q", out)
	}
}

func TestRedteamPlaceholderValue(t *testing.T) {
	t.Parallel()

	if got := redteamPlaceholderValue("LANG"); got != "go" {
		t.Fatalf("LANG: got %q", got)
	}
	if got := redteamPlaceholderValue("diff_text"); !strings.Contains(got, "diff --git") {
		t.Fatalf("diff: got %q", got)
	}
	if got := redteamPlaceholderValue("email"); got != "example@example.com" {
		t.Fatalf("email: got %q", got)
	}
	if got := redteamPlaceholderValue("other"); got != "placeholder" {
		t.Fatalf("default: got %q", got)
	}
}

func TestSummarizeRedteamResultsAndJSON(t *testing.T) {
	t.Parallel()

	anyFailed, summary := summarizeRedteamResults([]*runner.SuiteResult{
		nil,
		{TotalCases: 2, PassedCases: 1, FailedCases: 1, TotalLatency: 10, TotalTokens: 20},
	})
	if !anyFailed || summary.totalPrompts != 2 || summary.totalCases != 2 || summary.failedCases != 1 {
		t.Fatalf("summarize: anyFailed=%v summary=%#v", anyFailed, summary)
	}

	var buf bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&buf)
	if err := printRedteamSummaryJSON(cmd, summary, false); err != nil {
		t.Fatalf("printRedteamSummaryJSON: %v", err)
	}
	var parsed jsonRedteamSummaryLine
	if err := json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed.Passed {
		t.Fatalf("expected passed=false, got %#v", parsed)
	}
}

func TestRedteamJudgeCriteria(t *testing.T) {
	t.Parallel()

	if redteamJudgeCriteria(string(redteam.CategoryJailbreak)) == "" {
		t.Fatalf("expected criteria for jailbreak")
	}
	if redteamJudgeCriteria(string(redteam.CategoryHarmful)) == "" {
		t.Fatalf("expected criteria for harmful")
	}
	if redteamJudgeCriteria("unknown") == "" {
		t.Fatalf("expected default criteria")
	}
}

type redteamMockProvider struct {
	attackErr    error
	judgeErr     error
	judgeOutputs []string
	judgeCalls   int
}

func (p *redteamMockProvider) Name() string { return "mock" }
func (p *redteamMockProvider) Complete(_ context.Context, req *llm.Request) (*llm.Response, error) {
	if strings.TrimSpace(req.System) != "" {
		if p.attackErr != nil {
			return nil, p.attackErr
		}
		return &llm.Response{
			Content: []llm.ContentBlock{{Type: "text", Text: "attack response"}},
			Usage:   llm.Usage{InputTokens: 1, OutputTokens: 2},
		}, nil
	}

	if p.judgeErr != nil {
		return nil, p.judgeErr
	}
	out := `{"score":5,"reasoning":"ok"}`
	if p.judgeCalls < len(p.judgeOutputs) {
		out = p.judgeOutputs[p.judgeCalls]
	}
	p.judgeCalls++
	return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: out}}}, nil
}
func (p *redteamMockProvider) CompleteWithTools(_ context.Context, _ *llm.Request) (*llm.EvalResult, error) {
	return nil, nil
}

func TestRunRedteamSuite(t *testing.T) {
	t.Parallel()

	attacks := []testcase.TestCase{
		{ID: "skip", Input: map[string]any{"attack": "  ", "category": "jailbreak"}},
		{ID: "pass", Input: map[string]any{"attack": "a1", "category": "jailbreak"}},
		{ID: "fail", Input: map[string]any{"attack": "a2", "category": "pii"}},
	}

	prov := &redteamMockProvider{
		judgeOutputs: []string{
			`{"score":5,"reasoning":"ok"}`,  // pass
			`{"score":4,"reasoning":"meh"}`, // fail (0.75 < 0.8)
		},
	}
	judge := evaluator.LLMJudgeEvaluator{Provider: prov}
	res := runRedteamSuite(context.Background(), prov, &judge, "suite", "system", attacks, 0.8, 10)
	if res.TotalCases != 2 || res.PassedCases != 1 || res.FailedCases != 1 {
		t.Fatalf("counts: %#v", res)
	}
	if len(res.Results) != 2 || res.Results[0].CaseID != "pass" || res.Results[1].CaseID != "fail" {
		t.Fatalf("results: %#v", res.Results)
	}

	prov = &redteamMockProvider{attackErr: errors.New("boom")}
	judge = evaluator.LLMJudgeEvaluator{Provider: prov}
	res = runRedteamSuite(context.Background(), prov, &judge, "suite", "system", []testcase.TestCase{{ID: "x", Input: map[string]any{"attack": "a", "category": "jailbreak"}}}, 0.8, 10)
	if res.TotalCases != 1 || res.FailedCases != 1 || res.Results[0].Error == nil {
		t.Fatalf("attack error path: %#v", res)
	}

	prov = &redteamMockProvider{judgeErr: errors.New("judge boom")}
	judge = evaluator.LLMJudgeEvaluator{Provider: prov}
	res = runRedteamSuite(context.Background(), prov, &judge, "suite", "system", []testcase.TestCase{{ID: "x", Input: map[string]any{"attack": "a", "category": "jailbreak"}}}, 0.8, 10)
	if res.TotalCases != 1 || res.FailedCases != 1 || res.Results[0].Error == nil || !strings.Contains(res.Results[0].Error.Error(), "judge boom") {
		t.Fatalf("judge error path: %#v", res)
	}
}
