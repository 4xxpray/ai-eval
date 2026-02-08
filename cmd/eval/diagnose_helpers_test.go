package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

type noopProvider struct{}

func (noopProvider) Name() string { return "noop" }
func (noopProvider) Complete(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return nil, errors.New("noop")
}
func (noopProvider) CompleteWithTools(_ context.Context, _ *llm.Request) (*llm.EvalResult, error) {
	return nil, errors.New("noop")
}

func TestNewRunnerFromConfig(t *testing.T) {
	t.Parallel()

	if _, err := newRunnerFromConfig(nil, &config.Config{}); err == nil {
		t.Fatalf("expected error for nil provider")
	}
	if _, err := newRunnerFromConfig(noopProvider{}, nil); err == nil {
		t.Fatalf("expected error for nil config")
	}

	cfg := &config.Config{Evaluation: config.EvaluationConfig{Threshold: -0.1}}
	if _, err := newRunnerFromConfig(noopProvider{}, cfg); err == nil {
		t.Fatalf("expected error for invalid threshold")
	}

	cfg = &config.Config{Evaluation: config.EvaluationConfig{Trials: 0, Threshold: 0.6, Concurrency: 0}}
	r, err := newRunnerFromConfig(noopProvider{}, cfg)
	if err != nil || r == nil {
		t.Fatalf("newRunnerFromConfig: r=%v err=%v", r, err)
	}
}

func TestLoadPromptInput_FromFiles(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	yamlPath := filepath.Join(dir, "p.yaml")
	if err := os.WriteFile(yamlPath, []byte("name: example\ntemplate: hello\nis_system_prompt: true\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	in, err := loadPromptInput(yamlPath)
	if err != nil {
		t.Fatalf("loadPromptInput(yaml): %v", err)
	}
	if !in.IsYAML || in.Prompt == nil || in.PromptText != "hello" || in.NameHint != "example" {
		t.Fatalf("loadPromptInput(yaml): got %#v", in)
	}
	if in.SystemHint == nil || *in.SystemHint != true {
		t.Fatalf("loadPromptInput(yaml): expected SystemHint=true, got %#v", in.SystemHint)
	}

	textPath := filepath.Join(dir, "p.txt")
	if err := os.WriteFile(textPath, []byte("prompt text"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	in, err = loadPromptInput(textPath)
	if err != nil {
		t.Fatalf("loadPromptInput(text): %v", err)
	}
	if in.IsYAML || in.Prompt != nil || strings.TrimSpace(in.PromptText) != "prompt text" {
		t.Fatalf("loadPromptInput(text): got %#v", in)
	}

	if _, err := loadPromptInput(filepath.Join(dir, "missing.txt")); err == nil {
		t.Fatalf("expected error for missing file")
	}
	emptyPath := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(emptyPath, []byte(" \n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := loadPromptInput(emptyPath); err == nil {
		t.Fatalf("expected error for empty prompt")
	}
}

func TestLoadPromptInput_FromStdin(t *testing.T) {
	// Not parallel: mutates os.Stdin.
	dir := t.TempDir()
	_ = dir

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Pipe: %v", err)
	}
	old := os.Stdin
	os.Stdin = r
	t.Cleanup(func() { os.Stdin = old })
	t.Cleanup(func() { _ = r.Close() })

	if _, err := w.Write([]byte("stdin prompt")); err != nil {
		_ = w.Close()
		t.Fatalf("Write: %v", err)
	}
	_ = w.Close()

	in, err := loadPromptInput("")
	if err != nil {
		t.Fatalf("loadPromptInput(stdin): %v", err)
	}
	if strings.TrimSpace(in.PromptText) != "stdin prompt" || in.SourceLabel != "stdin" || in.NameHint != "prompt" {
		t.Fatalf("loadPromptInput(stdin): got %#v", in)
	}
}

func TestLoadTestSuites_FileAndDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	filePath := filepath.Join(dir, "suite.yaml")
	if err := os.WriteFile(filePath, []byte("suite: s\nprompt: p\ncases:\n  - id: c1\n    input: {}\n    expected:\n      exact_match: ok\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	suites, err := loadTestSuites(filePath)
	if err != nil || len(suites) != 1 {
		t.Fatalf("loadTestSuites(file): suites=%v err=%v", suites, err)
	}

	dirPath := filepath.Join(dir, "suites")
	if err := os.Mkdir(dirPath, 0o755); err != nil {
		t.Fatalf("Mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dirPath, "a.yaml"), []byte("suite: a\nprompt: p\ncases:\n  - id: a1\n    input: {}\n    expected:\n      exact_match: ok\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dirPath, "b.yaml"), []byte("suite: b\nprompt: p\ncases:\n  - id: b1\n    input: {}\n    expected:\n      exact_match: ok\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	suites, err = loadTestSuites(dirPath)
	if err != nil || len(suites) != 2 {
		t.Fatalf("loadTestSuites(dir): suites=%v err=%v", suites, err)
	}

	if _, err := loadTestSuites(filepath.Join(dir, "missing")); err == nil {
		t.Fatalf("expected error for missing suites path")
	}
}

func TestSelectSuitesAndPromptHints(t *testing.T) {
	t.Parallel()

	mkSuite := func(suite, promptName string, isSystem bool) *testcase.TestSuite {
		return &testcase.TestSuite{
			Suite:          suite,
			Prompt:         promptName,
			IsSystemPrompt: isSystem,
			Cases:          []testcase.TestCase{{ID: "c1", Input: map[string]any{}, Expected: testcase.Expected{ExactMatch: "ok"}}},
		}
	}

	if _, _, _, err := selectSuitesAndPromptHints(nil, []*testcase.TestSuite{mkSuite("s", "p", false)}); err == nil {
		t.Fatalf("expected error for nil prompt input")
	}
	if _, _, _, err := selectSuitesAndPromptHints(&promptInput{PromptText: "x"}, nil); err == nil {
		t.Fatalf("expected error for no suites")
	}
	if _, _, _, err := selectSuitesAndPromptHints(&promptInput{PromptText: "x"}, []*testcase.TestSuite{{Suite: "s"}}); err == nil {
		t.Fatalf("expected error for missing prompt reference")
	}

	yamlIn := &promptInput{IsYAML: true, PromptText: "x", Prompt: &prompt.Prompt{Name: "p1"}}
	if _, _, _, err := selectSuitesAndPromptHints(yamlIn, []*testcase.TestSuite{mkSuite("s1", "p1", false), mkSuite("s2", "p2", false)}); err == nil {
		t.Fatalf("expected error for multiple prompt names with YAML prompt")
	}
	if _, _, _, err := selectSuitesAndPromptHints(yamlIn, []*testcase.TestSuite{mkSuite("s1", "p2", false)}); err == nil {
		t.Fatalf("expected error for prompt name mismatch")
	}

	plainIn := &promptInput{IsYAML: false, PromptText: "x"}
	if _, _, _, err := selectSuitesAndPromptHints(plainIn, []*testcase.TestSuite{mkSuite("s1", "p1", false), mkSuite("s2", "p2", false)}); err == nil {
		t.Fatalf("expected error for multiple prompt names with plain prompt")
	}

	if _, _, _, err := selectSuitesAndPromptHints(plainIn, []*testcase.TestSuite{mkSuite("s1", "p1", false), mkSuite("s2", "p1", true)}); err == nil {
		t.Fatalf("expected error for mixed is_system_prompt")
	}

	sys := true
	yamlIn.SystemHint = &sys
	if _, _, _, err := selectSuitesAndPromptHints(yamlIn, []*testcase.TestSuite{mkSuite("s1", "p1", false)}); err == nil {
		t.Fatalf("expected error for is_system_prompt mismatch")
	}

	sys = false
	yamlIn.SystemHint = &sys
	suites, promptName, isSystem, err := selectSuitesAndPromptHints(yamlIn, []*testcase.TestSuite{
		mkSuite("b", "p1", false),
		nil,
		mkSuite("a", "p1", false),
	})
	if err != nil {
		t.Fatalf("selectSuitesAndPromptHints: %v", err)
	}
	if promptName != "p1" || isSystem {
		t.Fatalf("promptName/isSystem: got %q/%v", promptName, isSystem)
	}
	if len(suites) != 2 || strings.TrimSpace(suites[0].Suite) != "a" {
		t.Fatalf("suite sort/compact: got %#v", suites)
	}
}

func TestBuildPromptForRunAndOutputHelpers(t *testing.T) {
	t.Parallel()

	pIn := &promptInput{
		IsYAML:     true,
		PromptText: "text",
		Prompt: &prompt.Prompt{
			Name:           "old",
			Template:       "t",
			IsSystemPrompt: false,
		},
	}
	p := buildPromptForRun(pIn, "new", true)
	if p.Name != "new" || !p.IsSystemPrompt || p.Template != "t" {
		t.Fatalf("buildPromptForRun(yaml): got %#v", p)
	}

	out := buildDiagnoseJSONOutput(nil, nil, []*runner.SuiteResult{nil, &runner.SuiteResult{Suite: "s", TotalCases: 1, PassedCases: 1}}, &optimizer.DiagnoseResult{})
	if len(out.Suites) != 1 || out.Suites[0].Suite != "s" {
		t.Fatalf("buildDiagnoseJSONOutput: got %#v", out)
	}

	var buf bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&buf)
	printDiagnoseText(cmd, &prompt.Prompt{Name: "p"}, nil, []*runner.SuiteResult{&runner.SuiteResult{Suite: "s"}}, nil)
	if !strings.Contains(buf.String(), "Diagnosis: <nil>") {
		t.Fatalf("printDiagnoseText(nil): got %q", buf.String())
	}
}

func TestRunSuites(t *testing.T) {
	t.Parallel()

	if _, err := runSuites(context.Background(), nil, &prompt.Prompt{}, nil); err == nil || !strings.Contains(err.Error(), "nil runner") {
		t.Fatalf("expected nil runner error, got %v", err)
	}

	r, err := newRunnerFromConfig(noopProvider{}, &config.Config{Evaluation: config.EvaluationConfig{Threshold: 0.6}})
	if err != nil || r == nil {
		t.Fatalf("newRunnerFromConfig: r=%v err=%v", r, err)
	}

	if _, err := runSuites(context.Background(), r, nil, nil); err == nil || !strings.Contains(err.Error(), "nil prompt") {
		t.Fatalf("expected nil prompt error, got %v", err)
	}

	got, err := runSuites(context.Background(), r, &prompt.Prompt{Name: "p", Template: "t"}, []*testcase.TestSuite{nil})
	if err != nil {
		t.Fatalf("runSuites(nil suite): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results, got %#v", got)
	}

	if _, err := runSuites(nil, r, &prompt.Prompt{Name: "p", Template: "t"}, []*testcase.TestSuite{{Suite: "s"}}); err == nil {
		t.Fatalf("expected error for nil ctx")
	}
}
