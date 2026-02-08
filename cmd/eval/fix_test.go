package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
)

type mockLLMProvider struct {
	name     string
	response string
	err      error
	lastReq  *llm.Request
	nilResp  bool
}

func (m *mockLLMProvider) Name() string { return m.name }
func (m *mockLLMProvider) Complete(_ context.Context, req *llm.Request) (*llm.Response, error) {
	m.lastReq = req
	if m.err != nil {
		return nil, m.err
	}
	if m.nilResp {
		return nil, nil
	}
	return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: m.response}}}, nil
}
func (m *mockLLMProvider) CompleteWithTools(_ context.Context, _ *llm.Request) (*llm.EvalResult, error) {
	return nil, nil
}

func TestExtractRewritePrompt(t *testing.T) {
	t.Parallel()

	if got := extractRewritePrompt(nil); got != "" {
		t.Fatalf("extractRewritePrompt(nil): got %q", got)
	}

	diag := &optimizer.DiagnoseResult{
		Suggestions: []optimizer.FixSuggestion{
			{ID: "S1", Type: "rewrite_prompt", After: "fixed"},
		},
	}
	if got := extractRewritePrompt(diag); got != "fixed" {
		t.Fatalf("extractRewritePrompt(rewrite_prompt): got %q", got)
	}

	diag = &optimizer.DiagnoseResult{
		Suggestions: []optimizer.FixSuggestion{
			{ID: "S1", Type: "rewrite", After: "fixed2"},
		},
	}
	if got := extractRewritePrompt(diag); got != "fixed2" {
		t.Fatalf("extractRewritePrompt(rewrite): got %q", got)
	}
}

func TestRewritePromptFallback(t *testing.T) {
	t.Parallel()

	if _, err := rewritePromptFallback(context.Background(), nil, "p", &optimizer.DiagnoseResult{}); err == nil {
		t.Fatalf("expected error for nil provider")
	}
	if _, err := rewritePromptFallback(context.Background(), &mockLLMProvider{name: "m"}, "   ", &optimizer.DiagnoseResult{}); err == nil {
		t.Fatalf("expected error for empty prompt")
	}
	if _, err := rewritePromptFallback(context.Background(), &mockLLMProvider{name: "m"}, "p", nil); err == nil {
		t.Fatalf("expected error for nil diagnosis")
	}

	if _, err := rewritePromptFallback(context.Background(), &mockLLMProvider{name: "m", err: errors.New("boom")}, "p", &optimizer.DiagnoseResult{}); err == nil || !strings.Contains(err.Error(), "fix: llm:") {
		t.Fatalf("expected llm error, got %v", err)
	}
	if _, err := rewritePromptFallback(context.Background(), &mockLLMProvider{name: "m", nilResp: true}, "p", &optimizer.DiagnoseResult{}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("expected nil response error, got %v", err)
	}
	if _, err := rewritePromptFallback(context.Background(), &mockLLMProvider{name: "m", response: "not json"}, "p", &optimizer.DiagnoseResult{}); err == nil || !strings.Contains(err.Error(), "parse response") {
		t.Fatalf("expected parse error, got %v", err)
	}

	prov := &mockLLMProvider{name: "m", response: `{"fixed_prompt":"  fixed  "}`}
	got, err := rewritePromptFallback(context.Background(), prov, "orig", &optimizer.DiagnoseResult{RootCauses: []string{"x"}})
	if err != nil {
		t.Fatalf("rewritePromptFallback: %v", err)
	}
	if got != "fixed" {
		t.Fatalf("rewritePromptFallback: got %q want %q", got, "fixed")
	}
	if prov.lastReq == nil || len(prov.lastReq.Messages) != 1 || !strings.Contains(prov.lastReq.Messages[0].Content, "orig") {
		t.Fatalf("expected request to include prompt text, got %#v", prov.lastReq)
	}
}

func TestWriteFixedPrompt(t *testing.T) {
	t.Parallel()

	if err := writeFixedPrompt(nil, "x"); err == nil {
		t.Fatalf("expected error for nil prompt input")
	}
	if err := writeFixedPrompt(&promptInput{}, "x"); err == nil {
		t.Fatalf("expected error for missing path")
	}
	if err := writeFixedPrompt(&promptInput{Path: "x"}, " "); err == nil {
		t.Fatalf("expected error for empty fixed prompt")
	}

	dir := t.TempDir()

	yamlPath := filepath.Join(dir, "p.yaml")
	pIn := &promptInput{
		Path:   yamlPath,
		IsYAML: true,
		Prompt: &prompt.Prompt{
			Name:     "p",
			Template: "old",
		},
	}
	if err := writeFixedPrompt(pIn, "new"); err != nil {
		t.Fatalf("writeFixedPrompt(yaml): %v", err)
	}
	b, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !strings.Contains(string(b), "new") {
		t.Fatalf("expected yaml to contain new template, got %q", string(b))
	}

	if err := writeFixedPrompt(&promptInput{Path: filepath.Join(dir, "bad.yaml"), IsYAML: true, Prompt: nil}, "x"); err == nil {
		t.Fatalf("expected error for nil yaml prompt")
	}

	if err := writeFixedPrompt(&promptInput{Path: filepath.Join(dir, "looks.yaml"), IsYAML: false}, "x"); err == nil {
		t.Fatalf("expected refusal to overwrite yaml-looking file")
	}

	txtPath := filepath.Join(dir, "p.txt")
	if err := writeFixedPrompt(&promptInput{Path: txtPath, IsYAML: false}, "fixed"); err != nil {
		t.Fatalf("writeFixedPrompt(text): %v", err)
	}
	b, err = os.ReadFile(txtPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(b) != "fixed" {
		t.Fatalf("unexpected text output: %q", string(b))
	}
}
