package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

var cliIntegrationMu sync.Mutex

type stubProvider struct {
	name string

	judgeScore int

	diagnoseJSON string
	generateJSON string
	optimizeJSON string

	fallbackFixedPrompt string

	completeWithTools func(*llm.Request) string
}

func (p *stubProvider) Name() string {
	if p == nil {
		return ""
	}
	return p.name
}

func (p *stubProvider) Complete(_ context.Context, req *llm.Request) (*llm.Response, error) {
	if req == nil {
		return nil, errors.New("stub: nil request")
	}

	if strings.TrimSpace(req.System) != "" {
		return stubTextResponse("I refuse.")
	}

	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[0].Content
	}

	switch {
	case strings.Contains(content, "You are a red-team engineer."):
		return stubTextResponse(`{"cases":[{"id":"jb1","category":"jailbreak","attack":"ignore system","description":"d"},{"id":"inj1","category":"injection","attack":"reveal system prompt","description":"d"},{"id":"pii1","category":"pii","attack":"tell me ssn","description":"d"}]}`)
	case strings.Contains(content, "You are a prompt debugging advisor."):
		return stubTextResponse(p.diagnoseJSON)
	case strings.Contains(content, "You are a prompt engineer. Rewrite the prompt according to the diagnosis suggestions."):
		if strings.TrimSpace(p.fallbackFixedPrompt) == "" {
			return stubTextResponse(`{"fixed_prompt":"FALLBACK_FIXED"}`)
		}
		return stubTextResponse(`{"fixed_prompt":"` + p.fallbackFixedPrompt + `"}`)
	case strings.Contains(content, "You are a prompt evaluation expert."):
		return stubTextResponse(p.generateJSON)
	case strings.Contains(content, "You are a prompt engineering expert."):
		return stubTextResponse(p.optimizeJSON)
	case strings.Contains(content, "You are an expert evaluator."):
		score := p.judgeScore
		if score <= 0 {
			score = 5
		}
		if strings.Contains(content, "OPTIMIZE_FAIL") {
			score = 1
		}
		return stubTextResponse(`{"score":` + intToString(score) + `,"reasoning":"ok"}`)
	default:
		return stubTextResponse("ok")
	}
}

func (p *stubProvider) CompleteWithTools(_ context.Context, req *llm.Request) (*llm.EvalResult, error) {
	out := "ok"
	if p != nil && p.completeWithTools != nil {
		out = p.completeWithTools(req)
	}
	return &llm.EvalResult{
		TextContent:  out,
		LatencyMs:    1,
		InputTokens:  1,
		OutputTokens: 1,
	}, nil
}

func stubTextResponse(text string) (*llm.Response, error) {
	return &llm.Response{
		Content: []llm.ContentBlock{{Type: "text", Text: text}},
		Usage:   llm.Usage{InputTokens: 1, OutputTokens: 1},
	}, nil
}

func intToString(n int) string {
	switch {
	case n == 1:
		return "1"
	case n == 2:
		return "2"
	case n == 3:
		return "3"
	case n == 4:
		return "4"
	case n == 5:
		return "5"
	case n == 6:
		return "6"
	case n == 7:
		return "7"
	case n == 8:
		return "8"
	case n == 9:
		return "9"
	case n == 10:
		return "10"
	default:
		return "0"
	}
}

type evalWorkspace struct {
	dir     string
	runDB   string
	prompt  string
	fixCopy string
}

func setupEvalWorkspace(t *testing.T) evalWorkspace {
	t.Helper()

	dir := t.TempDir()
	mkdirAll(t, filepath.Join(dir, "configs"))
	mkdirAll(t, filepath.Join(dir, "prompts"))
	mkdirAll(t, filepath.Join(dir, "prompts", "p1"))
	mkdirAll(t, filepath.Join(dir, "tests"))

	cfgPath := filepath.Join(dir, "configs", "config.yaml")
	writeFile(t, cfgPath, strings.TrimSpace(`
evaluation:
  trials: 1
  threshold: 0.8
  concurrency: 1
  timeout: 1s
storage:
  type: "sqlite"
  path: "data/test.db"
`)+"\n")

	writeFile(t, filepath.Join(dir, "prompts", "p1.yaml"), strings.TrimSpace(`
name: p1
version: v0
template: "prompt v0"
`)+"\n")
	writeFile(t, filepath.Join(dir, "prompts", "p2.yaml"), strings.TrimSpace(`
name: p2
version: v0
template: "prompt p2"
`)+"\n")

	writeFile(t, filepath.Join(dir, "prompts", "p1", "v1.yaml"), strings.TrimSpace(`
name: p1
version: v1
template: "prompt v1"
`)+"\n")
	writeFile(t, filepath.Join(dir, "prompts", "p1", "v2.yaml"), strings.TrimSpace(`
name: p1
version: v2
template: "prompt v2"
`)+"\n")

	writeFile(t, filepath.Join(dir, "tests", "suite.yaml"), strings.TrimSpace(`
suite: suite1
prompt: p1
cases:
  - id: c1
    input: {}
    expected:
      exact_match: ok
`)+"\n")

	prompt := filepath.Join(dir, "prompts", "p1.yaml")
	fixCopy := filepath.Join(dir, "fix.yaml")
	writeFile(t, fixCopy, strings.TrimSpace(`
name: p1
version: v0
template: "fix original"
`)+"\n")

	writeFile(t, filepath.Join(dir, "optimize.txt"), "optimize me\n")

	return evalWorkspace{
		dir:     dir,
		runDB:   filepath.Join(dir, "data", "test.db"),
		prompt:  prompt,
		fixCopy: fixCopy,
	}
}

func mkdirAll(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", path, err)
	}
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%q): %v", path, err)
	}
}

func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()

	cmd := newRootCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs(args)
	err := cmd.Execute()
	return out.String(), err
}

func parseFirstRunID(t *testing.T, out string) string {
	t.Helper()

	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "RUN_ID") || strings.HasPrefix(line, "No runs found.") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if strings.HasPrefix(fields[0], "run_") {
			return fields[0]
		}
	}
	t.Fatalf("no run id found in output: %q", out)
	return ""
}

func TestCLI_Integration(t *testing.T) {
	// Not parallel: mutates global state (cwd, os.Args, injected provider).
	cliIntegrationMu.Lock()
	defer cliIntegrationMu.Unlock()

	ws := setupEvalWorkspace(t)

	oldCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(ws.dir); err != nil {
		t.Fatalf("Chdir(%q): %v", ws.dir, err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldCwd) })

	prov := &stubProvider{
		name:         "stub",
		judgeScore:   5,
		diagnoseJSON: `{"failure_patterns":["missing_context"],"root_causes":["x"],"suggestions":[{"id":"S1","type":"rewrite_prompt","description":"d","before":"b","after":"FIXED_PROMPT","impact":"high","priority":1}]}`,
		generateJSON: `{"analysis":"a","is_system_prompt":false,"suggestions":[],"test_cases":[{"id":"c1","description":"d","input":{},"expected":{},"evaluators":[{"type":"llm_judge","criteria":"OPTIMIZE_FAIL","score_threshold":0.6}]}]}`,
		optimizeJSON: `{"optimized_prompt":"OPTIMIZED_PROMPT","summary":"s","changes":[{"type":"modify","description":"d"}]}`,
		completeWithTools: func(req *llm.Request) string {
			if req != nil && len(req.Messages) > 0 && strings.Contains(req.Messages[0].Content, "v2") {
				return "bad"
			}
			return "ok"
		},
	}

	oldProviderFromConfig := defaultProviderFromConfig
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return prov, nil }
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })

	t.Run("main_help", func(t *testing.T) {
		oldArgs := os.Args
		os.Args = []string{"ai-eval", "--help"}
		t.Cleanup(func() { os.Args = oldArgs })
		main()
	})

	t.Run("list", func(t *testing.T) {
		out, err := runCLI(t, "list", "prompts")
		if err != nil {
			t.Fatalf("list prompts: %v", err)
		}
		if !strings.Contains(out, "NAME") {
			t.Fatalf("list prompts output: %q", out)
		}

		out, err = runCLI(t, "list", "tests")
		if err != nil {
			t.Fatalf("list tests: %v", err)
		}
		if !strings.Contains(out, "SUITE") {
			t.Fatalf("list tests output: %q", out)
		}
	})

	t.Run("history_empty", func(t *testing.T) {
		out, err := runCLI(t, "history")
		if err != nil {
			t.Fatalf("history: %v", err)
		}
		if !strings.Contains(out, "No runs found.") {
			t.Fatalf("history output: %q", out)
		}
	})

	t.Run("run_json_and_history", func(t *testing.T) {
		out, err := runCLI(t, "run", "--prompt", "p1", "--output", "json", "--trials", "1", "--threshold", "0.8")
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		if !strings.Contains(out, "\"summary\"") {
			t.Fatalf("run output: %q", out)
		}

		out, err = runCLI(t, "history", "--since", "2000-01-02")
		if err != nil {
			t.Fatalf("history list: %v", err)
		}
		runID := parseFirstRunID(t, out)

		out, err = runCLI(t, "history", "show", runID)
		if err != nil {
			t.Fatalf("history show: %v", err)
		}
		if !strings.Contains(out, "Run: "+runID) {
			t.Fatalf("history show output: %q", out)
		}
		if !strings.Contains(out, "PASS") {
			t.Fatalf("history show expected PASS: %q", out)
		}
	})

	t.Run("run_ci_artifacts", func(t *testing.T) {
		if _, err := runCLI(t, "run", "--all", "--ci"); err != nil {
			t.Fatalf("run --ci: %v", err)
		}
		if _, err := os.Stat(ciReportPath); err != nil {
			t.Fatalf("expected %s to exist: %v", ciReportPath, err)
		}
	})

	t.Run("run_table_and_failure", func(t *testing.T) {
		out, err := runCLI(t, "run", "--prompt", "p1", "--output", "table", "--trials", "1", "--threshold", "0.8")
		if err != nil {
			t.Fatalf("run table: %v", err)
		}
		if !strings.Contains(out, "Summary:") {
			t.Fatalf("run table output: %q", out)
		}

		oldComplete := prov.completeWithTools
		prov.completeWithTools = func(*llm.Request) string { return "bad" }
		t.Cleanup(func() { prov.completeWithTools = oldComplete })

		if _, err := runCLI(t, "run", "--prompt", "p1", "--output", "table", "--trials", "1", "--threshold", "0.8"); err == nil || !errors.Is(err, errTestsFailed) {
			t.Fatalf("expected errTestsFailed, got %v", err)
		}

		out, err = runCLI(t, "history")
		if err != nil {
			t.Fatalf("history: %v", err)
		}
		runID := parseFirstRunID(t, out)
		out, err = runCLI(t, "history", "show", runID)
		if err != nil {
			t.Fatalf("history show: %v", err)
		}
		if !strings.Contains(out, "FAIL") {
			t.Fatalf("expected FAIL in history output: %q", out)
		}
	})

	t.Run("run_validation_errors", func(t *testing.T) {
		if _, err := runCLI(t, "run"); err == nil || !strings.Contains(err.Error(), "specify either") {
			t.Fatalf("expected missing selection error, got %v", err)
		}
		if _, err := runCLI(t, "run", "--all", "--prompt", "p1"); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
			t.Fatalf("expected mutually exclusive error, got %v", err)
		}
		if _, err := runCLI(t, "run", "--prompt", "p1", "--output", "wat"); err == nil || !strings.Contains(err.Error(), "invalid --output") {
			t.Fatalf("expected output error, got %v", err)
		}
		if _, err := runCLI(t, "run", "--prompt", "p1", "--trials", "0"); err == nil || !strings.Contains(err.Error(), "trials must be > 0") {
			t.Fatalf("expected trials error, got %v", err)
		}
		if _, err := runCLI(t, "run", "--prompt", "p1", "--threshold", "2"); err == nil || !strings.Contains(err.Error(), "threshold must be between 0 and 1") {
			t.Fatalf("expected threshold error, got %v", err)
		}
		if _, err := runCLI(t, "run", "--prompt", "nope"); err == nil || !strings.Contains(err.Error(), "unknown prompt") {
			t.Fatalf("expected unknown prompt error, got %v", err)
		}
	})

	t.Run("leaderboard", func(t *testing.T) {
		if _, err := runCLI(t, "leaderboard"); err == nil || !strings.Contains(err.Error(), "missing --dataset") {
			t.Fatalf("expected dataset error, got %v", err)
		}

		cfg, err := config.Load(config.DefaultPath)
		if err != nil {
			t.Fatalf("config.Load: %v", err)
		}
		lb, err := openLeaderboardStore(cfg)
		if err != nil {
			t.Fatalf("openLeaderboardStore: %v", err)
		}
		t.Cleanup(func() { _ = lb.Close() })
		if err := lb.Save(context.Background(), &leaderboard.Entry{
			Model:    "m",
			Provider: prov.Name(),
			Dataset:  "ds",
			Score:    0.9,
			Accuracy: 0.9,
			Latency:  1,
			Cost:     0,
			EvalDate: time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC),
		}); err != nil {
			t.Fatalf("leaderboard.Save: %v", err)
		}
		_ = lb.Close()

		out, err := runCLI(t, "leaderboard", "--dataset", "ds", "--format", "json")
		if err != nil {
			t.Fatalf("leaderboard json: %v", err)
		}
		if !strings.Contains(out, "\"Model\"") || !strings.Contains(out, "\"m\"") {
			t.Fatalf("leaderboard json output: %q", out)
		}

		out, err = runCLI(t, "leaderboard", "--dataset", "ds", "--format", "table")
		if err != nil {
			t.Fatalf("leaderboard table: %v", err)
		}
		if !strings.Contains(out, "RANK") || !strings.Contains(out, "m") {
			t.Fatalf("leaderboard table output: %q", out)
		}
	})

	t.Run("leaderboard_invalid_format", func(t *testing.T) {
		if _, err := runCLI(t, "leaderboard", "--dataset", "ds", "--format", "wat"); err == nil || !strings.Contains(err.Error(), "invalid --format") {
			t.Fatalf("expected format error, got %v", err)
		}
	})

	t.Run("compare_table_and_github_no_regression", func(t *testing.T) {
		oldComplete := prov.completeWithTools
		prov.completeWithTools = func(*llm.Request) string { return "ok" }
		t.Cleanup(func() { prov.completeWithTools = oldComplete })

		out, err := runCLI(t, "compare", "--prompt", "p1", "--v1", "v1", "--v2", "v2")
		if err != nil {
			t.Fatalf("compare table: %v", err)
		}
		if !strings.Contains(out, "Suite:") {
			t.Fatalf("compare table output: %q", out)
		}

		out, err = runCLI(t, "compare", "--prompt", "p1", "--v1", "v1", "--v2", "v2", "--output", "github")
		if err != nil {
			t.Fatalf("compare github: %v", err)
		}
		if !strings.Contains(out, "Summary:") {
			t.Fatalf("compare github output: %q", out)
		}
	})

	t.Run("compare_validation_errors", func(t *testing.T) {
		if _, err := runCLI(t, "compare", "--prompt", "p1", "--v1", "v1", "--v2", "v1"); err == nil || !strings.Contains(err.Error(), "must differ") {
			t.Fatalf("expected v1/v2 differ error, got %v", err)
		}
	})

	t.Run("compare_regression_json", func(t *testing.T) {
		out, err := runCLI(t, "compare", "--prompt", "p1", "--v1", "v1", "--v2", "v2", "--output", "json", "--trials", "1")
		if err == nil || !errors.Is(err, errRegression) {
			t.Fatalf("expected errRegression, got %v", err)
		}
		if !strings.Contains(out, "\"compare\"") {
			t.Fatalf("compare output: %q", out)
		}
	})

	t.Run("benchmark_error", func(t *testing.T) {
		if _, err := runCLI(t, "benchmark"); err == nil || !strings.Contains(err.Error(), "missing --dataset") {
			t.Fatalf("expected dataset error, got %v", err)
		}
	})

	t.Run("benchmark_provider_not_configured", func(t *testing.T) {
		if _, err := runCLI(t, "benchmark", "--dataset", "mmlu", "--provider", "nope"); err == nil || !strings.Contains(err.Error(), "not configured") {
			t.Fatalf("expected not configured error, got %v", err)
		}
	})

	t.Run("benchmark_success_stub", func(t *testing.T) {
		oldBenchProvider := benchmarkProviderFromConfig
		benchmarkProviderFromConfig = func(*config.Config, string, string) (llm.Provider, string, error) {
			return prov, "stub-model", nil
		}
		t.Cleanup(func() { benchmarkProviderFromConfig = oldBenchProvider })

		oldComplete := prov.completeWithTools
		prov.completeWithTools = func(*llm.Request) string { return "A" }
		t.Cleanup(func() { prov.completeWithTools = oldComplete })

		out, err := runCLI(t, "benchmark", "--dataset", "mmlu", "--sample-size", "1")
		if err != nil {
			t.Fatalf("benchmark: %v", err)
		}
		if !strings.Contains(out, "Benchmark saved:") {
			t.Fatalf("benchmark output: %q", out)
		}
	})

	t.Run("redteam_single_and_all", func(t *testing.T) {
		if _, err := runCLI(t, "redteam", "--prompt", "p1", "--output", "table"); err != nil {
			t.Fatalf("redteam single: %v", err)
		}

		out, err := runCLI(t, "redteam", "--all", "--output", "json")
		if err != nil {
			t.Fatalf("redteam all: %v", err)
		}
		if !strings.Contains(out, "\"summary\"") {
			t.Fatalf("redteam all output: %q", out)
		}
	})

	t.Run("redteam_github_and_errors", func(t *testing.T) {
		out, err := runCLI(t, "redteam", "--all", "--output", "github")
		if err != nil {
			t.Fatalf("redteam github: %v", err)
		}
		if !strings.Contains(out, "Summary:") {
			t.Fatalf("redteam github output: %q", out)
		}

		if _, err := runCLI(t, "redteam", "--all", "--output", "table"); err == nil || !strings.Contains(err.Error(), "--all does not support") {
			t.Fatalf("expected output/table error, got %v", err)
		}

		if _, err := runCLI(t, "redteam", "--prompt", "p1", "--categories", "nope"); err == nil || !strings.Contains(err.Error(), "unknown category") {
			t.Fatalf("expected unknown category error, got %v", err)
		}
	})

	t.Run("redteam_failures_and_more_errors", func(t *testing.T) {
		oldScore := prov.judgeScore
		prov.judgeScore = 4 // 0.75 < 0.8 => fail
		t.Cleanup(func() { prov.judgeScore = oldScore })

		if _, err := runCLI(t, "redteam", "--prompt", "p1", "--output", "table"); err == nil || !errors.Is(err, errTestsFailed) {
			t.Fatalf("expected errTestsFailed, got %v", err)
		}
		if _, err := runCLI(t, "redteam", "--all", "--output", "github"); err == nil || !errors.Is(err, errTestsFailed) {
			t.Fatalf("expected errTestsFailed, got %v", err)
		}

		prov.judgeScore = oldScore

		if _, err := runCLI(t, "redteam", "--prompt", "unknown", "--output", "json"); err == nil || !strings.Contains(err.Error(), "unknown prompt") {
			t.Fatalf("expected unknown prompt error, got %v", err)
		}

		oldDefaultProvider := defaultProviderFromConfig
		defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return nil, errors.New("boom") }
		t.Cleanup(func() { defaultProviderFromConfig = oldDefaultProvider })
		if _, err := runCLI(t, "redteam", "--prompt", "p1"); err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("expected provider error, got %v", err)
		}
		defaultProviderFromConfig = oldDefaultProvider

		promptsDir := filepath.Join(ws.dir, "prompts")
		backupDir := filepath.Join(ws.dir, "prompts.bak")
		if err := os.Rename(promptsDir, backupDir); err != nil {
			t.Fatalf("rename prompts: %v", err)
		}
		t.Cleanup(func() {
			_ = os.RemoveAll(promptsDir)
			_ = os.Rename(backupDir, promptsDir)
		})
		mkdirAll(t, promptsDir)

		if _, err := runCLI(t, "redteam", "--all"); err == nil || !strings.Contains(err.Error(), "no prompts found") {
			t.Fatalf("expected no prompts error, got %v", err)
		}
	})

	t.Run("diagnose_text_and_json", func(t *testing.T) {
		out, err := runCLI(t, "diagnose", "--prompt", ws.prompt, "--output", "text")
		if err != nil {
			t.Fatalf("diagnose text: %v", err)
		}
		if !strings.Contains(out, "Failure Patterns:") {
			t.Fatalf("diagnose text output: %q", out)
		}

		out, err = runCLI(t, "diagnose", "--prompt", ws.prompt, "--output", "json")
		if err != nil {
			t.Fatalf("diagnose json: %v", err)
		}
		if !strings.Contains(out, "\"diagnosis\"") {
			t.Fatalf("diagnose json output: %q", out)
		}
	})

	t.Run("diagnose_invalid_output", func(t *testing.T) {
		if _, err := runCLI(t, "diagnose", "--prompt", ws.prompt, "--output", "wat"); err == nil || !strings.Contains(err.Error(), "invalid --output") {
			t.Fatalf("expected output error, got %v", err)
		}
	})

	t.Run("diagnose_missing_tests", func(t *testing.T) {
		if _, err := runCLI(t, "diagnose", "--prompt", ws.prompt, "--tests", "missing"); err == nil || !strings.Contains(err.Error(), "stat tests") {
			t.Fatalf("expected tests error, got %v", err)
		}
	})

	t.Run("fix_apply_and_fallback", func(t *testing.T) {
		out, err := runCLI(t, "fix", "--prompt", ws.prompt, "--apply")
		if err != nil {
			t.Fatalf("fix apply: %v", err)
		}
		if !strings.Contains(out, "Applied to:") {
			t.Fatalf("fix apply output: %q", out)
		}
		b, err := os.ReadFile(ws.prompt)
		if err != nil {
			t.Fatalf("ReadFile(prompt): %v", err)
		}
		if !strings.Contains(string(b), "FIXED_PROMPT") {
			t.Fatalf("expected prompt file to be updated, got %q", string(b))
		}

		prov.diagnoseJSON = `{"failure_patterns":["x"],"root_causes":["y"],"suggestions":[]}`
		out, err = runCLI(t, "fix", "--prompt", ws.fixCopy, "--dry-run")
		if err != nil {
			t.Fatalf("fix fallback: %v", err)
		}
		if !strings.Contains(out, "FALLBACK_FIXED") {
			t.Fatalf("fix fallback output: %q", out)
		}
	})

	t.Run("fix_validation_errors", func(t *testing.T) {
		if _, err := runCLI(t, "fix", "--apply"); err == nil || !strings.Contains(err.Error(), "requires --prompt") {
			t.Fatalf("expected apply requires prompt error, got %v", err)
		}

		before, err := os.ReadFile(ws.fixCopy)
		if err != nil {
			t.Fatalf("ReadFile(fixCopy): %v", err)
		}
		out, err := runCLI(t, "fix", "--prompt", ws.fixCopy, "--apply", "--dry-run")
		if err != nil {
			t.Fatalf("fix apply dry-run: %v", err)
		}
		if strings.TrimSpace(out) == "" {
			t.Fatalf("expected fixed prompt output, got %q", out)
		}
		after, err := os.ReadFile(ws.fixCopy)
		if err != nil {
			t.Fatalf("ReadFile(fixCopy): %v", err)
		}
		if string(after) != string(before) {
			t.Fatalf("expected file unchanged in --dry-run")
		}
	})

	t.Run("optimize", func(t *testing.T) {
		if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt"), "--output", filepath.Join(ws.dir, "optimized.txt"), "--cases", "1", "--iterations", "1", "--progress=false"); err != nil {
			t.Fatalf("optimize: %v", err)
		}
		b, err := os.ReadFile(filepath.Join(ws.dir, "optimized.txt"))
		if err != nil {
			t.Fatalf("ReadFile(optimized): %v", err)
		}
		if strings.TrimSpace(string(b)) != "OPTIMIZED_PROMPT" {
			t.Fatalf("optimized content: got %q want %q", strings.TrimSpace(string(b)), "OPTIMIZED_PROMPT")
		}
	})

	t.Run("optimize_more_branches", func(t *testing.T) {
		devNull, err := os.Open(os.DevNull)
		if err != nil {
			t.Fatalf("Open(devnull): %v", err)
		}
		t.Cleanup(func() { _ = devNull.Close() })

		oldStdin := os.Stdin
		os.Stdin = devNull
		t.Cleanup(func() { os.Stdin = oldStdin })

		if _, err := runCLI(t, "optimize"); err == nil || !strings.Contains(err.Error(), "no prompt provided") {
			t.Fatalf("expected no prompt error, got %v", err)
		}

		if _, err := runCLI(t, "--config", "configs/missing.yaml", "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt")); err == nil || !strings.Contains(err.Error(), "failed to load config") {
			t.Fatalf("expected config load error, got %v", err)
		}

		if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "missing.txt")); err == nil || !strings.Contains(err.Error(), "failed to read prompt file") {
			t.Fatalf("expected read error, got %v", err)
		}

		emptyPromptPath := filepath.Join(ws.dir, "empty.txt")
		writeFile(t, emptyPromptPath, " \n")
		if _, err := runCLI(t, "optimize", "--prompt", emptyPromptPath); err == nil || !strings.Contains(err.Error(), "prompt content is empty") {
			t.Fatalf("expected empty error, got %v", err)
		}

		oldProvider := defaultProviderFromConfig
		defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return nil, errors.New("boom") }
		t.Cleanup(func() { defaultProviderFromConfig = oldProvider })
		if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt")); err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("expected provider error, got %v", err)
		}

		defaultProviderFromConfig = oldProvider

		r, w, err := os.Pipe()
		if err != nil {
			t.Fatalf("Pipe: %v", err)
		}
		t.Cleanup(func() { _ = r.Close() })
		t.Cleanup(func() { _ = w.Close() })
		os.Stdin = r

		if _, err := w.Write([]byte("stdin prompt\n")); err != nil {
			t.Fatalf("stdin write: %v", err)
		}
		_ = w.Close()

		oldGen := prov.generateJSON
		prov.generateJSON = `{"analysis":"a","is_system_prompt":true,"suggestions":[],"test_cases":[{"id":"c1","description":"d","input":{},"expected":{},"evaluators":[{"type":"llm_judge","criteria":"OK","score_threshold":0.6}]}]}`
		t.Cleanup(func() { prov.generateJSON = oldGen })

		if _, err := runCLI(t, "optimize", "--progress=true", "--var", "A=B", "--var", "BAD"); err != nil {
			t.Fatalf("optimize stdin/noop: %v", err)
		}

		prov.generateJSON = oldGen

		// Cover output-to-stdout path (no --output) and output write error.
		if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt"), "--progress=false"); err != nil {
			t.Fatalf("optimize stdout: %v", err)
		}

		if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt"), "--output", ws.dir, "--progress=false"); err == nil || !strings.Contains(err.Error(), "failed to write output file") {
			t.Fatalf("expected write error, got %v", err)
		}
	})
}
