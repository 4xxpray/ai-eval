package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad_ReadError(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err == nil {
		t.Fatalf("Load: expected error")
	}
	if !strings.Contains(err.Error(), "config: read") {
		t.Fatalf("error: got %q", err)
	}
}

func TestLoad_ParseError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.yaml")
	if err := os.WriteFile(path, []byte(":"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := Load(path)
	if err == nil {
		t.Fatalf("Load: expected error")
	}
	if !strings.Contains(err.Error(), "config: parse") {
		t.Fatalf("error: got %q", err)
	}
}

func TestLoad_DefaultPathDefaultsAndEnvOverrides(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "configs"), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}

	cfgPath := filepath.Join(dir, DefaultPath)
	if err := os.WriteFile(cfgPath, []byte(strings.TrimSpace(`
llm:
  default_provider: "  "
  providers:
    claude:
      api_key: "file_key"
      base_url: "https://example.test"
      model: "m1"
evaluation:
  trials: 1
  threshold: 0.5
storage:
  type: memory
`)), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWD) })

	t.Setenv("ANTHROPIC_API_KEY", "env_key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "env_token_ignored")
	t.Setenv("OPENAI_API_KEY", "openai_env_key")

	cfg, err := Load(" \t ")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg == nil {
		t.Fatalf("Load: nil cfg")
	}
	if cfg.LLM.Providers == nil {
		t.Fatalf("Providers: nil")
	}
	if got := cfg.LLM.DefaultProvider; got != "claude" {
		t.Fatalf("DefaultProvider: got %q want %q", got, "claude")
	}

	cp := cfg.LLM.Providers["claude"]
	if cp.APIKey != "env_key" {
		t.Fatalf("claude api_key: got %q want %q", cp.APIKey, "env_key")
	}
	if cp.BaseURL != "https://example.test" || cp.Model != "m1" {
		t.Fatalf("claude other fields changed: got base_url=%q model=%q", cp.BaseURL, cp.Model)
	}

	op := cfg.LLM.Providers["openai"]
	if op.APIKey != "openai_env_key" {
		t.Fatalf("openai api_key: got %q want %q", op.APIKey, "openai_env_key")
	}
}

func TestLoad_ProvidersInitAndDefaults_NoEnv(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(path, []byte(strings.TrimSpace(`
llm: {}
evaluation:
  trials: 1
  threshold: 0.5
`)), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("OPENAI_API_KEY", "")

	cfg, err := Load(" \t " + path + " \n")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg == nil {
		t.Fatalf("Load: nil cfg")
	}
	if cfg.LLM.Providers == nil {
		t.Fatalf("Providers: nil")
	}
	if got := cfg.LLM.DefaultProvider; got != "claude" {
		t.Fatalf("DefaultProvider: got %q want %q", got, "claude")
	}
	if len(cfg.LLM.Providers) != 0 {
		t.Fatalf("Providers len: got %d want %d", len(cfg.LLM.Providers), 0)
	}
}

func TestLoad_AnthropicAuthTokenFallback(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(path, []byte(strings.TrimSpace(`
llm:
  providers:
    claude:
      api_key: "file_key"
      model: "m1"
evaluation:
  trials: 1
  threshold: 0.5
`)), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "token_key")
	t.Setenv("OPENAI_API_KEY", "")

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg == nil {
		t.Fatalf("Load: nil cfg")
	}
	cp := cfg.LLM.Providers["claude"]
	if cp.APIKey != "token_key" {
		t.Fatalf("claude api_key: got %q want %q", cp.APIKey, "token_key")
	}
	if cp.Model != "m1" {
		t.Fatalf("claude model changed: got %q want %q", cp.Model, "m1")
	}
}
