package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

type errWriter struct {
	err error
}

func (w errWriter) Write([]byte) (int, error) {
	return 0, w.err
}

func TestOpenLeaderboardStore_Defaults(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	dir := t.TempDir()
	oldCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldCwd) })
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	cfg := &config.Config{}
	st, err := openLeaderboardStore(cfg)
	if err != nil {
		t.Fatalf("openLeaderboardStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
}

func TestBuildPromptForRun_NonYAML(t *testing.T) {
	t.Parallel()

	pIn := &promptInput{
		IsYAML:     false,
		Prompt:     nil,
		PromptText: "hi",
	}
	got := buildPromptForRun(pIn, "p1", true)
	if got == nil {
		t.Fatalf("expected non-nil prompt")
	}
	if got.Name != "p1" || got.Template != "hi" || !got.IsSystemPrompt {
		t.Fatalf("unexpected prompt: %#v", got)
	}
}

func TestCompactSuites_Empty(t *testing.T) {
	t.Parallel()

	if got := compactSuites(nil); got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}
	if got := compactSuites([]*testcase.TestSuite{}); len(got) != 0 {
		t.Fatalf("expected empty, got %#v", got)
	}
}

func TestSelectSuitesAndPromptHints_CompactsToEmpty(t *testing.T) {
	t.Parallel()

	_, _, _, err := selectSuitesAndPromptHints(&promptInput{}, []*testcase.TestSuite{nil})
	if err == nil || !strings.Contains(err.Error(), "no test suites loaded") {
		t.Fatalf("expected no test suites loaded error, got %v", err)
	}
}

func TestPrintRunJSON_EncodeErrors(t *testing.T) {
	t.Parallel()

	t.Run("suite_line", func(t *testing.T) {
		cmd := &cobra.Command{}
		cmd.SetOut(errWriter{err: errors.New("boom")})

		runs := []app.SuiteRun{{PromptName: "p1"}}
		if err := printRunJSON(cmd, runs, app.RunSummary{}); err == nil || !strings.Contains(err.Error(), "run: marshal json") {
			t.Fatalf("expected marshal json error, got %v", err)
		}
	})

	t.Run("summary_line", func(t *testing.T) {
		cmd := &cobra.Command{}
		cmd.SetOut(errWriter{err: errors.New("boom")})

		if err := printRunJSON(cmd, nil, app.RunSummary{}); err == nil || !strings.Contains(err.Error(), "run: marshal json") {
			t.Fatalf("expected marshal json error, got %v", err)
		}
	})
}

func TestPrintRedteamSummaryJSON_EncodeError(t *testing.T) {
	t.Parallel()

	cmd := &cobra.Command{}
	cmd.SetOut(errWriter{err: errors.New("boom")})

	if err := printRedteamSummaryJSON(cmd, redteamSummary{totalPrompts: 1}, true); err == nil || !strings.Contains(err.Error(), "redteam: marshal json") {
		t.Fatalf("expected marshal json error, got %v", err)
	}
}

func TestOptimizeCmd_ReadStdinError(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "cfg.yaml")
	writeFile(t, cfgPath, "{}\n")

	st := &cliState{configPath: cfgPath}
	cmd := newOptimizeCmd(st)
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})

	oldStdin := os.Stdin
	t.Cleanup(func() { os.Stdin = oldStdin })

	dirFile, err := os.Open(dir)
	if err != nil {
		t.Fatalf("Open(dir): %v", err)
	}
	t.Cleanup(func() { _ = dirFile.Close() })
	os.Stdin = dirFile

	if err := cmd.RunE(cmd, nil); err == nil || !strings.Contains(err.Error(), "failed to read from stdin") {
		t.Fatalf("expected stdin read error, got %v", err)
	}
}

func TestRunCompare_CoverUncoveredErrors(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	st := &cliState{
		cfg: &config.Config{
			Evaluation: config.EvaluationConfig{Trials: 1, Threshold: 0.8, Concurrency: 1},
		},
	}

	t.Run("find_v1_error", func(t *testing.T) {
		dir := t.TempDir()
		oldCwd, err := os.Getwd()
		if err != nil {
			t.Fatalf("Getwd: %v", err)
		}
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })
		if err := os.Chdir(dir); err != nil {
			t.Fatalf("Chdir: %v", err)
		}

		mkdirAll(t, defaultPromptsDir)
		writeFile(t, filepath.Join(defaultPromptsDir, "p.yaml"), "name: other\nversion: v1\ntemplate: hi\n")

		cmd := &cobra.Command{}
		cmd.SetOut(&bytes.Buffer{})

		opts := &compareOptions{promptName: "missing", v1: "v1", v2: "v2", trials: -1}
		if err := runCompare(cmd, st, opts); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("find_v2_error", func(t *testing.T) {
		dir := t.TempDir()
		oldCwd, err := os.Getwd()
		if err != nil {
			t.Fatalf("Getwd: %v", err)
		}
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })
		if err := os.Chdir(dir); err != nil {
			t.Fatalf("Chdir: %v", err)
		}

		mkdirAll(t, defaultPromptsDir)
		writeFile(t, filepath.Join(defaultPromptsDir, "v1.yaml"), "name: p1\nversion: v1\ntemplate: hi\n")

		cmd := &cobra.Command{}
		cmd.SetOut(&bytes.Buffer{})

		opts := &compareOptions{promptName: "p1", v1: "v1", v2: "v2", trials: -1}
		if err := runCompare(cmd, st, opts); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("load_tests_error", func(t *testing.T) {
		dir := t.TempDir()
		oldCwd, err := os.Getwd()
		if err != nil {
			t.Fatalf("Getwd: %v", err)
		}
		t.Cleanup(func() { _ = os.Chdir(oldCwd) })
		if err := os.Chdir(dir); err != nil {
			t.Fatalf("Chdir: %v", err)
		}

		mkdirAll(t, defaultPromptsDir)
		writeFile(t, filepath.Join(defaultPromptsDir, "v1.yaml"), "name: p1\nversion: v1\ntemplate: hi\n")
		writeFile(t, filepath.Join(defaultPromptsDir, "v2.yaml"), "name: p1\nversion: v2\ntemplate: hi\n")

		cmd := &cobra.Command{}
		cmd.SetOut(&bytes.Buffer{})

		opts := &compareOptions{promptName: "p1", v1: "v1", v2: "v2", trials: -1}
		if err := runCompare(cmd, st, opts); err == nil {
			t.Fatalf("expected error")
		}
	})
}

func TestRunFix_LoadTestSuitesError(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	oldProviderFromConfig := defaultProviderFromConfig
	t.Cleanup(func() { defaultProviderFromConfig = oldProviderFromConfig })
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return &stubProvider{name: "stub"}, nil }

	dir := t.TempDir()
	promptPath := filepath.Join(dir, "p.txt")
	writeFile(t, promptPath, "hi\n")

	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})

	st := &cliState{cfg: &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.8}}}
	opts := &fixOptions{promptPath: promptPath, testsPath: filepath.Join(dir, "missing.yaml")}
	if err := runFix(cmd, st, opts); err == nil || !strings.Contains(err.Error(), "stat tests") {
		t.Fatalf("expected tests stat error, got %v", err)
	}
}
