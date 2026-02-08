package main

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

func TestConfigLoadErrorInCmdPreRuns(t *testing.T) {
	t.Parallel()

	st := &cliState{configPath: "configs/does-not-exist.yaml"}
	tests := []struct {
		name string
		pre  func() error
	}{
		{
			name: "run",
			pre: func() error {
				cmd := newRunCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "compare",
			pre: func() error {
				cmd := newCompareCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "redteam",
			pre: func() error {
				cmd := newRedteamCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "diagnose",
			pre: func() error {
				cmd := newDiagnoseCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "fix",
			pre: func() error {
				cmd := newFixCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "benchmark",
			pre: func() error {
				cmd := newBenchmarkCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "leaderboard",
			pre: func() error {
				cmd := newLeaderboardCmd(st)
				return cmd.PreRunE(cmd, nil)
			},
		},
		{
			name: "history",
			pre: func() error {
				cmd := newHistoryCmd(st)
				return cmd.PersistentPreRunE(cmd, nil)
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.pre(); err == nil {
				t.Fatalf("expected config load error")
			}
		})
	}
}

func TestRunDiagnose_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})

	if err := runDiagnose(cmd, nil, &diagnoseOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runDiagnose(cmd, &cliState{}, &diagnoseOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runDiagnose(cmd, &cliState{cfg: &config.Config{}}, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil options error, got %v", err)
	}

	st := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.8, Concurrency: 1}}}

	if err := runDiagnose(cmd, st, &diagnoseOptions{output: "wat"}); err == nil || !strings.Contains(err.Error(), "invalid --output") {
		t.Fatalf("expected invalid output error, got %v", err)
	}

	oldProviderFromConfig := defaultProviderFromConfig
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return nil, errors.New("boom")
	}
	if err := runDiagnose(cmd, st, &diagnoseOptions{output: " "}); err == nil || !strings.Contains(err.Error(), "diagnose: boom") {
		t.Fatalf("expected provider error wrapper, got %v", err)
	}

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return nil, nil
	}
	if err := runDiagnose(cmd, st, &diagnoseOptions{output: "text"}); err == nil || !strings.Contains(err.Error(), "nil llm provider") {
		t.Fatalf("expected nil provider error, got %v", err)
	}

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return noopProvider{}, nil
	}
	if err := runDiagnose(cmd, st, &diagnoseOptions{promptPath: filepath.Join(t.TempDir(), "missing.txt")}); err == nil || !strings.Contains(err.Error(), "read prompt") {
		t.Fatalf("expected loadPromptInput error, got %v", err)
	}

	dir := t.TempDir()
	promptPath := filepath.Join(dir, "p.yaml")
	writeFile(t, promptPath, "name: p1\ntemplate: hi\n")
	testsPath := filepath.Join(dir, "suite.yaml")
	writeFile(t, testsPath, "suite: s\nprompt: p2\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")

	if err := runDiagnose(cmd, st, &diagnoseOptions{promptPath: promptPath, testsPath: testsPath, output: "text"}); err == nil || !strings.Contains(err.Error(), "prompt name mismatch") {
		t.Fatalf("expected prompt mismatch error, got %v", err)
	}
}

func TestLoadPromptInput_YAML_Errors(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	badYAMLPath := filepath.Join(dir, "bad.yaml")
	writeFile(t, badYAMLPath, "name: [\n")
	if _, err := loadPromptInput(badYAMLPath); err == nil || !strings.Contains(err.Error(), "load prompt") {
		t.Fatalf("expected load prompt error, got %v", err)
	}

	emptyTemplatePath := filepath.Join(dir, "empty_template.yaml")
	writeFile(t, emptyTemplatePath, "name: x\nversion: v1\n")
	if _, err := loadPromptInput(emptyTemplatePath); err == nil || !strings.Contains(err.Error(), "empty template") {
		t.Fatalf("expected empty template error, got %v", err)
	}

	noNamePath := filepath.Join(dir, "noname.yaml")
	writeFile(t, noNamePath, "version: v1\ntemplate: hi\n")
	in, err := loadPromptInput(noNamePath)
	if err != nil {
		t.Fatalf("loadPromptInput(noname): %v", err)
	}
	if in.NameHint != "noname" {
		t.Fatalf("expected NameHint fallback %q, got %q", "noname", in.NameHint)
	}
}

func TestLoadPromptInput_Stdin_Errors(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	oldStdin := os.Stdin
	t.Cleanup(func() { os.Stdin = oldStdin })

	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatalf("Open(devnull): %v", err)
	}
	t.Cleanup(func() { _ = devNull.Close() })
	os.Stdin = devNull
	if _, err := loadPromptInput(""); err == nil || !strings.Contains(err.Error(), "no prompt provided") {
		t.Fatalf("expected no prompt provided error, got %v", err)
	}

	os.Stdin = oldStdin
	dir := t.TempDir()
	dirFile, err := os.Open(dir)
	if err != nil {
		t.Fatalf("Open(dir): %v", err)
	}
	t.Cleanup(func() { _ = dirFile.Close() })
	os.Stdin = dirFile
	if _, err := loadPromptInput(""); err == nil || !strings.Contains(err.Error(), "read stdin") {
		t.Fatalf("expected read stdin error, got %v", err)
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Pipe: %v", err)
	}
	t.Cleanup(func() { _ = r.Close() })
	os.Stdin = r
	_, _ = w.Write([]byte(" \n"))
	_ = w.Close()
	if _, err := loadPromptInput(""); err == nil || !strings.Contains(err.Error(), "content is empty") {
		t.Fatalf("expected empty content error, got %v", err)
	}
}

func TestLoadTestSuites_DefaultAndErrors(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	oldCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldCwd) })

	dir := t.TempDir()
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	mkdirAll(t, filepath.Join(dir, defaultTestsDir))
	writeFile(t, filepath.Join(dir, defaultTestsDir, "s.yaml"), "suite: s\nprompt: p\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")

	suites, err := loadTestSuites("")
	if err != nil || len(suites) != 1 {
		t.Fatalf("loadTestSuites(default): suites=%v err=%v", suites, err)
	}

	badDir := filepath.Join(dir, "badtests")
	mkdirAll(t, badDir)
	writeFile(t, filepath.Join(badDir, "bad.yaml"), "suite: s\nprompt: p\ncases: []\n")
	if _, err := loadTestSuites(badDir); err == nil || !strings.Contains(err.Error(), "load tests dir") {
		t.Fatalf("expected load tests dir error, got %v", err)
	}

	badFile := filepath.Join(dir, "badfile.yaml")
	writeFile(t, badFile, "suite: s\nprompt: p\ncases: []\n")
	if _, err := loadTestSuites(badFile); err == nil || !strings.Contains(err.Error(), "load tests file") {
		t.Fatalf("expected load tests file error, got %v", err)
	}
}

func TestPrintDiagnoseText_EmptyLists(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&buf)

	printDiagnoseText(cmd,
		&prompt.Prompt{Name: "p"},
		nil,
		[]*runner.SuiteResult{nil, &runner.SuiteResult{Suite: "s", TotalCases: 1, PassedCases: 1, FailedCases: 0, PassRate: 1, AvgScore: 1}},
		&optimizer.DiagnoseResult{},
	)
	out := buf.String()
	if !strings.Contains(out, "- (none)") {
		t.Fatalf("expected empty sections marker, got %q", out)
	}
}

func TestRunEvaluations_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	oldProviderFromConfig := defaultProviderFromConfig
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return &stubProvider{name: "stub"}, nil }

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetContext(context.Background())

	if err := runEvaluations(cmd, nil, &runOptions{}); err == nil || !strings.Contains(err.Error(), "nil state") {
		t.Fatalf("expected nil state error, got %v", err)
	}
	if err := runEvaluations(cmd, &cliState{}, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil options error, got %v", err)
	}
	if err := runEvaluations(cmd, &cliState{}, &runOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}

	mkWorkspace := func(t *testing.T) string {
		t.Helper()
		dir := t.TempDir()
		mkdirAll(t, filepath.Join(dir, defaultPromptsDir))
		mkdirAll(t, filepath.Join(dir, defaultTestsDir))
		return dir
	}
	writePrompt := func(t *testing.T, dir string, file string, name string) {
		t.Helper()
		writeFile(t, filepath.Join(dir, defaultPromptsDir, file), "name: "+name+"\nversion: v0\ntemplate: prompt\n")
	}
	writeSuite := func(t *testing.T, dir string, file string, suiteName string, promptName string) {
		t.Helper()
		writeFile(t, filepath.Join(dir, defaultTestsDir, file), "suite: "+suiteName+"\nprompt: "+promptName+"\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")
	}

	baseCfg := func() *config.Config {
		return &config.Config{
			Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.8, Concurrency: 0},
			Storage:    config.StorageConfig{Type: "memory"},
		}
	}

	t.Run("load_prompts_error", func(t *testing.T) {
		dir := t.TempDir()
		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "read dir") {
			t.Fatalf("expected load prompts error, got %v", err)
		}
	})

	t.Run("index_prompts_duplicate_name", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1a.yaml", "p1")
		writePrompt(t, dir, "p1b.yaml", "p1")
		writeSuite(t, dir, "s.yaml", "s1", "p1")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "duplicate prompt name") {
			t.Fatalf("expected index prompts error, got %v", err)
		}
	})

	t.Run("load_tests_error", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")
		_ = os.RemoveAll(filepath.Join(dir, defaultTestsDir))

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "read dir") {
			t.Fatalf("expected load tests error, got %v", err)
		}
	})

	t.Run("index_suites_unknown_prompt", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")
		writeSuite(t, dir, "s.yaml", "s1", "nope")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "references unknown prompt") {
			t.Fatalf("expected index suites error, got %v", err)
		}
	})

	t.Run("all_no_suites", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{all: true, trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "no test suites found") {
			t.Fatalf("expected no suites error, got %v", err)
		}
	})

	t.Run("prompt_no_suites", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "no test suites found for prompt") {
			t.Fatalf("expected no suites for prompt error, got %v", err)
		}
	})

	t.Run("provider_error", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")
		writeSuite(t, dir, "s.yaml", "s1", "p1")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return nil, errors.New("boom") }
		t.Cleanup(func() {
			defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return &stubProvider{name: "stub"}, nil }
		})

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "run: boom") {
			t.Fatalf("expected provider wrapper error, got %v", err)
		}
	})

	t.Run("sort_suites_comparator", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")
		writeSuite(t, dir, "b.yaml", "b", "p1")
		writeSuite(t, dir, "a.yaml", "a", "p1")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: baseCfg()}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err != nil {
			t.Fatalf("expected success, got %v", err)
		}
	})

	t.Run("save_run_error", func(t *testing.T) {
		dir := mkWorkspace(t)
		writePrompt(t, dir, "p1.yaml", "p1")
		writeSuite(t, dir, "s.yaml", "s1", "p1")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		cfg := baseCfg()
		cfg.Storage.Type = "nope"
		st := &cliState{cfg: cfg}
		if err := runEvaluations(cmd, st, &runOptions{promptName: "p1", output: "table", trials: -1, threshold: -1}); err == nil || !strings.Contains(err.Error(), "open store") {
			t.Fatalf("expected saveRunToStore error, got %v", err)
		}
	})
}

func TestRunCompare_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	oldProviderFromConfig := defaultProviderFromConfig
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return &stubProvider{name: "stub"}, nil }

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetContext(context.Background())

	st := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.8, Concurrency: 0}}}

	if err := runCompare(cmd, nil, &compareOptions{}); err == nil || !strings.Contains(err.Error(), "nil state") {
		t.Fatalf("expected nil state error, got %v", err)
	}
	if err := runCompare(cmd, st, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil opts error, got %v", err)
	}
	if err := runCompare(cmd, &cliState{}, &compareOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}

	if err := runCompare(cmd, st, &compareOptions{v1: "v1", v2: "v2"}); err == nil || !strings.Contains(err.Error(), "missing --prompt") {
		t.Fatalf("expected missing prompt error, got %v", err)
	}
	if err := runCompare(cmd, st, &compareOptions{promptName: "p1"}); err == nil || !strings.Contains(err.Error(), "missing --v1/--v2") {
		t.Fatalf("expected missing v1/v2 error, got %v", err)
	}
	if err := runCompare(cmd, st, &compareOptions{promptName: "p1", v1: "v1", v2: "v2", output: "wat"}); err == nil || !strings.Contains(err.Error(), "invalid --output") {
		t.Fatalf("expected invalid output error, got %v", err)
	}

	stBadTrials := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Trials: 0, Threshold: 0.8}}}
	if err := runCompare(cmd, stBadTrials, &compareOptions{promptName: "p1", v1: "v1", v2: "v2"}); err == nil || !strings.Contains(err.Error(), "trials must be > 0") {
		t.Fatalf("expected trials error, got %v", err)
	}

	stBadThreshold := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 2}}}
	if err := runCompare(cmd, stBadThreshold, &compareOptions{promptName: "p1", v1: "v1", v2: "v2", trials: -1}); err == nil || !strings.Contains(err.Error(), "threshold must be between 0 and 1") {
		t.Fatalf("expected threshold error, got %v", err)
	}

	t.Run("load_prompts_recursive_error", func(t *testing.T) {
		dir := t.TempDir()
		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		st := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.8, Concurrency: 0}}}
		if err := runCompare(cmd, st, &compareOptions{promptName: "p1", v1: "v1", v2: "v2", trials: -1}); err == nil {
			t.Fatalf("expected prompt load error")
		}
	})

	t.Run("no_matching_suites", func(t *testing.T) {
		dir := t.TempDir()
		mkdirAll(t, filepath.Join(dir, defaultPromptsDir))
		mkdirAll(t, filepath.Join(dir, defaultTestsDir))

		writeFile(t, filepath.Join(dir, defaultPromptsDir, "v1.yaml"), "name: p1\nversion: v1\ntemplate: v1\n")
		writeFile(t, filepath.Join(dir, defaultPromptsDir, "v2.yaml"), "name: p1\nversion: v2\ntemplate: v2\n")
		writeFile(t, filepath.Join(dir, defaultTestsDir, "s.yaml"), "suite: s\nprompt: p2\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		if err := runCompare(cmd, st, &compareOptions{promptName: "p1", v1: "v1", v2: "v2", trials: -1}); err == nil || !strings.Contains(err.Error(), "no test suites found") {
			t.Fatalf("expected no suites error, got %v", err)
		}
	})

	t.Run("provider_error", func(t *testing.T) {
		dir := t.TempDir()
		mkdirAll(t, filepath.Join(dir, defaultPromptsDir))
		mkdirAll(t, filepath.Join(dir, defaultTestsDir))

		writeFile(t, filepath.Join(dir, defaultPromptsDir, "v1.yaml"), "name: p1\nversion: v1\ntemplate: v1\n")
		writeFile(t, filepath.Join(dir, defaultPromptsDir, "v2.yaml"), "name: p1\nversion: v2\ntemplate: v2\n")
		writeFile(t, filepath.Join(dir, defaultTestsDir, "a.yaml"), "suite: b\nprompt: p1\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")
		writeFile(t, filepath.Join(dir, defaultTestsDir, "b.yaml"), "suite: a\nprompt: p1\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return nil, errors.New("boom") }
		t.Cleanup(func() {
			defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return &stubProvider{name: "stub"}, nil }
		})

		if err := runCompare(cmd, st, &compareOptions{promptName: "p1", v1: "v1", v2: "v2", trials: -1}); err == nil || !strings.Contains(err.Error(), "compare: boom") {
			t.Fatalf("expected provider error wrapper, got %v", err)
		}
	})
}

func TestRunFix_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetContext(context.Background())

	if err := runFix(cmd, nil, &fixOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runFix(cmd, &cliState{cfg: &config.Config{}}, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil opts error, got %v", err)
	}
	if err := runFix(cmd, &cliState{cfg: &config.Config{}}, &fixOptions{apply: true}); err == nil || !strings.Contains(err.Error(), "requires --prompt") {
		t.Fatalf("expected apply requires prompt error, got %v", err)
	}

	st := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.8, Concurrency: 1}}}

	oldProviderFromConfig := defaultProviderFromConfig
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return nil, errors.New("boom")
	}
	if err := runFix(cmd, st, &fixOptions{}); err == nil || !strings.Contains(err.Error(), "fix: boom") {
		t.Fatalf("expected provider wrapper error, got %v", err)
	}

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return nil, nil
	}
	if err := runFix(cmd, st, &fixOptions{}); err == nil || !strings.Contains(err.Error(), "nil llm provider") {
		t.Fatalf("expected nil provider error, got %v", err)
	}

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return &stubProvider{name: "stub", diagnoseJSON: "not json"}, nil
	}
	if err := runFix(cmd, st, &fixOptions{promptPath: filepath.Join(t.TempDir(), "missing.yaml"), testsPath: filepath.Join(t.TempDir(), "missing")}); err == nil {
		t.Fatalf("expected loadPromptInput error")
	}
}

func TestWriteFixedPrompt_WriteErrors(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	pIn := &promptInput{
		Path:   filepath.Join(dir, "out.yaml"),
		IsYAML: true,
		Prompt: &prompt.Prompt{
			Name:     "p",
			Template: "t",
		},
	}
	outDir := filepath.Join(dir, "dir.yaml")
	mkdirAll(t, outDir)
	pIn.Path = outDir
	if err := writeFixedPrompt(pIn, "new"); err == nil || !strings.Contains(err.Error(), "fix: write") {
		t.Fatalf("expected yaml write error, got %v", err)
	}

	txtDir := filepath.Join(dir, "dir.txt")
	mkdirAll(t, txtDir)
	if err := writeFixedPrompt(&promptInput{Path: txtDir, IsYAML: false}, "new"); err == nil || !strings.Contains(err.Error(), "fix: write") {
		t.Fatalf("expected text write error, got %v", err)
	}
}

func TestRunRedteam_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetContext(context.Background())

	st := &cliState{cfg: &config.Config{}}
	if err := runRedteam(cmd, nil, &redteamOptions{}); err == nil || !strings.Contains(err.Error(), "nil state") {
		t.Fatalf("expected nil state error, got %v", err)
	}
	if err := runRedteam(cmd, st, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil options error, got %v", err)
	}
	if err := runRedteam(cmd, &cliState{}, &redteamOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}

	oldProviderFromConfig := defaultProviderFromConfig
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return &stubProvider{name: "stub"}, nil }

	if err := runRedteam(cmd, st, &redteamOptions{all: true, promptName: "p1"}); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("expected mutual exclusion error, got %v", err)
	}
	if err := runRedteam(cmd, st, &redteamOptions{}); err == nil || !strings.Contains(err.Error(), "specify either") {
		t.Fatalf("expected selection error, got %v", err)
	}

	t.Run("load_prompts_error", func(t *testing.T) {
		dir := t.TempDir()
		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		if err := runRedteam(cmd, st, &redteamOptions{promptName: "p1"}); err == nil {
			t.Fatalf("expected load prompts error")
		}
	})

	t.Run("render_system_prompt_error", func(t *testing.T) {
		dir := t.TempDir()
		mkdirAll(t, filepath.Join(dir, defaultPromptsDir))
		mkdirAll(t, filepath.Join(dir, defaultTestsDir))

		writeFile(t, filepath.Join(dir, defaultPromptsDir, "p.yaml"), "name: p1\nversion: v0\ntemplate: \"{{\"\n")

		oldCwd, _ := os.Getwd()
		_ = os.Chdir(dir)
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })

		if err := runRedteam(cmd, st, &redteamOptions{promptName: "p1"}); err == nil || !strings.Contains(err.Error(), "unmatched") {
			t.Fatalf("expected render error, got %v", err)
		}
	})
}

func TestRunHistory_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})

	if err := runHistoryList(cmd, nil, &historyOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runHistoryList(cmd, &cliState{cfg: &config.Config{}}, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil options error, got %v", err)
	}
	if err := runHistoryList(cmd, &cliState{cfg: &config.Config{}}, &historyOptions{since: "nope"}); err == nil || !strings.Contains(err.Error(), "invalid --since") {
		t.Fatalf("expected since parse error, got %v", err)
	}

	stBadStore := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "nope"}}}
	cmd.SetContext(context.Background())
	if err := runHistoryList(cmd, stBadStore, &historyOptions{}); err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected open store error, got %v", err)
	}

	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel()
	cmd.SetContext(ctxCanceled)
	stMem := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "memory"}}}
	if err := runHistoryList(cmd, stMem, &historyOptions{}); err == nil {
		t.Fatalf("expected list runs error")
	}

	if err := runHistoryShow(cmd, nil, "x"); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runHistoryShow(cmd, stMem, " "); err == nil || !strings.Contains(err.Error(), "missing run id") {
		t.Fatalf("expected missing run id error, got %v", err)
	}
	if err := runHistoryShow(cmd, stBadStore, "x"); err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected open store error, got %v", err)
	}

	cmd.SetContext(ctxCanceled)
	if err := runHistoryShow(cmd, stMem, "x"); err == nil {
		t.Fatalf("expected GetRun error")
	}

	t.Run("no_suites_for_run", func(t *testing.T) {
		dir := t.TempDir()
		dbPath := filepath.Join(dir, "nosuites.sqlite")
		stor, err := store.NewSQLiteStore(dbPath)
		if err != nil {
			t.Fatalf("NewSQLiteStore: %v", err)
		}
		_ = stor.Close()

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("sql.Open: %v", err)
		}
		t.Cleanup(func() { _ = db.Close() })

		startedAt := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
		finishedAt := startedAt.Add(time.Second)
		_, err = db.Exec(`INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			"run_only",
			startedAt.UnixMilli(),
			finishedAt.UnixMilli(),
			0, 0, 0,
			`{}`,
		)
		if err != nil {
			t.Fatalf("insert run: %v", err)
		}
		_ = db.Close()

		cmd := &cobra.Command{}
		cmd.SetOut(&bytes.Buffer{})
		cmd.SetContext(context.Background())
		st := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "sqlite", Path: dbPath}}}
		if err := runHistoryShow(cmd, st, "run_only"); err != nil {
			t.Fatalf("expected show to succeed, got %v", err)
		}
	})

	t.Run("suite_results_parse_error", func(t *testing.T) {
		dir := t.TempDir()
		dbPath := filepath.Join(dir, "badcases.sqlite")
		stor, err := store.NewSQLiteStore(dbPath)
		if err != nil {
			t.Fatalf("NewSQLiteStore: %v", err)
		}
		_ = stor.Close()

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("sql.Open: %v", err)
		}
		t.Cleanup(func() { _ = db.Close() })

		startedAt := time.Date(2026, 2, 7, 0, 0, 0, 0, time.UTC)
		finishedAt := startedAt.Add(time.Second)
		_, err = db.Exec(`INSERT INTO runs (id, started_at, finished_at, total_suites, passed_suites, failed_suites, config_json) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			"run_bad",
			startedAt.UnixMilli(),
			finishedAt.UnixMilli(),
			1, 1, 0,
			`{}`,
		)
		if err != nil {
			t.Fatalf("insert run: %v", err)
		}
		_, err = db.Exec(`INSERT INTO suite_results (id, run_id, prompt_name, prompt_version, suite_name, total_cases, passed_cases, failed_cases, pass_rate, avg_score, total_latency, total_tokens, created_at, case_results) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			"run_bad_suite_1",
			"run_bad",
			"p1",
			"v1",
			"suite1",
			1, 1, 0,
			1.0, 1.0,
			0, 0,
			startedAt.UnixMilli(),
			[]byte("{not json"),
		)
		if err != nil {
			t.Fatalf("insert suite: %v", err)
		}
		_ = db.Close()

		cmd := &cobra.Command{}
		cmd.SetContext(context.Background())
		st := &cliState{cfg: &config.Config{Storage: config.StorageConfig{Type: "sqlite", Path: dbPath}}}
		if err := runHistoryShow(cmd, st, "run_bad"); err == nil {
			t.Fatalf("expected suite results parse error")
		}
	})
}

func TestRunOptimize_ShowProgressAndGenerateError(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

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
		diagnoseJSON: `{"failure_patterns":["x"],"root_causes":["y"],"suggestions":[]}`,
		generateJSON: `{"analysis":"a","is_system_prompt":false,"suggestions":[],"test_cases":[{"id":"c1","description":"d","input":{},"expected":{},"evaluators":[{"type":"llm_judge","criteria":"OPTIMIZE_FAIL","score_threshold":0.6}]}]}`,
		optimizeJSON: `{"optimized_prompt":"OPTIMIZED_PROMPT","summary":"s","changes":[{"type":"modify","description":"d"}]}`,
	}

	oldProviderFromConfig := defaultProviderFromConfig
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return prov, nil }
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })

	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatalf("Open(devnull): %v", err)
	}
	t.Cleanup(func() { _ = devNull.Close() })

	oldStdout := os.Stdout
	os.Stdout = devNull
	t.Cleanup(func() { os.Stdout = oldStdout })

	outPath := filepath.Join(ws.dir, "optimized_out.txt")
	if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt"), "--output", outPath, "--cases", "1", "--iterations", "1", "--progress=true"); err != nil {
		t.Fatalf("optimize showProgress: %v", err)
	}

	prov.generateJSON = "not json"
	if _, err := runCLI(t, "optimize", "--prompt", filepath.Join(ws.dir, "optimize.txt"), "--cases", "1", "--iterations", "1", "--progress=false"); err == nil || !strings.Contains(err.Error(), "failed to generate test cases") {
		t.Fatalf("expected generate error, got %v", err)
	}

	oldStdin := os.Stdin
	os.Stdin = devNull
	t.Cleanup(func() { os.Stdin = oldStdin })
	if _, err := runCLI(t, "optimize", "--progress=false"); err == nil || !strings.Contains(err.Error(), "no prompt provided") {
		t.Fatalf("expected no prompt error, got %v", err)
	}
}

func TestRunBenchmark_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})

	if err := runBenchmark(cmd, nil, &benchmarkOptions{}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runBenchmark(cmd, &cliState{cfg: &config.Config{}}, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil options error, got %v", err)
	}

	oldBenchProvider := benchmarkProviderFromConfig
	t.Cleanup(func() { benchmarkProviderFromConfig = oldBenchProvider })
	benchmarkProviderFromConfig = func(*config.Config, string, string) (llm.Provider, string, error) {
		return &stubProvider{name: "stub"}, "stub-model", nil
	}

	cfg := &config.Config{Storage: config.StorageConfig{Type: "nope"}}
	if err := runBenchmark(cmd, &cliState{cfg: cfg}, &benchmarkOptions{dataset: "mmlu"}); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("expected open leaderboard error, got %v", err)
	}

	dir := t.TempDir()
	cfg = &config.Config{Storage: config.StorageConfig{Type: "memory"}}
	st := &cliState{cfg: cfg}

	oldEnv := os.Getenv("AI_EVAL_MMLU_PATH")
	os.Setenv("AI_EVAL_MMLU_PATH", dir)
	t.Cleanup(func() { os.Setenv("AI_EVAL_MMLU_PATH", oldEnv) })

	if err := runBenchmark(cmd, st, &benchmarkOptions{dataset: "mmlu", sampleSize: 1}); err == nil {
		t.Fatalf("expected dataset load error")
	}

	os.Setenv("AI_EVAL_MMLU_PATH", filepath.Join(dir, "missing.jsonl"))

	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel()
	cmd.SetContext(ctxCanceled)
	if err := runBenchmark(cmd, st, &benchmarkOptions{dataset: "mmlu", sampleSize: 1}); err == nil {
		t.Fatalf("expected save error")
	}
}

func TestRunLeaderboard_ErrorPaths(t *testing.T) {
	t.Parallel()

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})

	if err := runLeaderboard(cmd, nil, &leaderboardOptions{dataset: "ds"}); err == nil || !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("expected missing config error, got %v", err)
	}
	if err := runLeaderboard(cmd, &cliState{cfg: &config.Config{}}, nil); err == nil || !strings.Contains(err.Error(), "nil options") {
		t.Fatalf("expected nil options error, got %v", err)
	}
}

func TestListTests_SortComparator(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	oldCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldCwd) })

	dir := t.TempDir()
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	mkdirAll(t, filepath.Join(dir, defaultTestsDir))
	writeFile(t, filepath.Join(dir, defaultTestsDir, "b.yaml"), "suite: b\nprompt: p\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")
	writeFile(t, filepath.Join(dir, defaultTestsDir, "a.yaml"), "suite: a\nprompt: p\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n")

	var buf bytes.Buffer
	c := newListTestsCmd()
	c.SetOut(&buf)
	if err := c.RunE(c, nil); err != nil {
		t.Fatalf("list tests: %v", err)
	}
}

func TestMain_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	ws := setupEvalWorkspace(t)

	oldCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(ws.dir); err != nil {
		t.Fatalf("Chdir(%q): %v", ws.dir, err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldCwd) })

	prov := &stubProvider{name: "stub"}
	oldProviderFromConfig := defaultProviderFromConfig
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return prov, nil }
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })

	var gotExit []int
	oldExit := osExit
	osExit = func(code int) { gotExit = append(gotExit, code) }
	t.Cleanup(func() { osExit = oldExit })

	var stderr bytes.Buffer
	oldStderr := stderrWriter
	stderrWriter = &stderr
	t.Cleanup(func() { stderrWriter = oldStderr })

	oldArgs := os.Args
	t.Cleanup(func() { os.Args = oldArgs })

	gotExit = nil
	stderr.Reset()
	os.Args = []string{"ai-eval", "run"}
	main()
	if len(gotExit) != 1 || gotExit[0] != 1 {
		t.Fatalf("expected exit=1, got %v", gotExit)
	}
	if stderr.Len() == 0 {
		t.Fatalf("expected stderr output for non-sentinel error")
	}

	prov.completeWithTools = func(*llm.Request) string { return "bad" }
	gotExit = nil
	stderr.Reset()
	os.Args = []string{"ai-eval", "run", "--prompt", "p1", "--output", "table", "--trials", "1", "--threshold", "0.8"}
	main()
	if len(gotExit) != 1 || gotExit[0] != 1 {
		t.Fatalf("expected exit=1, got %v", gotExit)
	}
	if stderr.Len() != 0 {
		t.Fatalf("expected no stderr for errTestsFailed, got %q", stderr.String())
	}

	prov.completeWithTools = func(req *llm.Request) string {
		if req != nil && len(req.Messages) > 0 && strings.Contains(req.Messages[0].Content, "v2") {
			return "bad"
		}
		return "ok"
	}
	gotExit = nil
	stderr.Reset()
	os.Args = []string{"ai-eval", "compare", "--prompt", "p1", "--v1", "v1", "--v2", "v2", "--output", "json", "--trials", "1"}
	main()
	if len(gotExit) != 1 || gotExit[0] != 1 {
		t.Fatalf("expected exit=1, got %v", gotExit)
	}
	if stderr.Len() != 0 {
		t.Fatalf("expected no stderr for errRegression, got %q", stderr.String())
	}
}
