package main

import (
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
)

func TestResolveBenchmarkDataset(t *testing.T) {
	t.Parallel()

	if _, err := resolveBenchmarkDataset("", 0); err == nil {
		t.Fatalf("expected error for missing dataset")
	}
	if _, err := resolveBenchmarkDataset("mmlu", -1); err == nil {
		t.Fatalf("expected error for negative sample size")
	}
	if _, err := resolveBenchmarkDataset("wat", 0); err == nil {
		t.Fatalf("expected error for unknown dataset")
	}

	if ds, err := resolveBenchmarkDataset("mmlu", 10); err != nil || ds == nil || ds.Name() != "mmlu" {
		t.Fatalf("mmlu: ds=%v err=%v", ds, err)
	}
	if ds, err := resolveBenchmarkDataset("humaneval", 1); err != nil || ds == nil || ds.Name() != "humaneval" {
		t.Fatalf("humaneval: ds=%v err=%v", ds, err)
	}
	if ds, err := resolveBenchmarkDataset("gsm8k", 1); err != nil || ds == nil || ds.Name() != "gsm8k" {
		t.Fatalf("gsm8k: ds=%v err=%v", ds, err)
	}
}

func TestNormalizeProvider(t *testing.T) {
	t.Parallel()

	if got := normalizeProvider(" anthropic "); got != "claude" {
		t.Fatalf("normalizeProvider(anthropic): got %q", got)
	}
	if got := normalizeProvider("OpenAI"); got != "openai" {
		t.Fatalf("normalizeProvider(openai): got %q", got)
	}
}

func TestResolveBenchmarkProvider(t *testing.T) {
	t.Parallel()

	if _, _, err := resolveBenchmarkProvider(nil, "", ""); err == nil {
		t.Fatalf("expected error for nil config")
	}

	cfg := &config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: " ",
			Providers:       map[string]config.ProviderConfig{},
		},
	}
	if _, _, err := resolveBenchmarkProvider(cfg, "", ""); err == nil {
		t.Fatalf("expected error for missing provider")
	}

	cfg = &config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: "claude",
			Providers: map[string]config.ProviderConfig{
				"openai": {APIKey: "k"},
			},
		},
	}
	if _, _, err := resolveBenchmarkProvider(cfg, "", ""); err == nil || !strings.Contains(err.Error(), "not configured") {
		t.Fatalf("expected not configured error, got %v", err)
	}

	cfg = &config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: "custom",
			Providers: map[string]config.ProviderConfig{
				"custom": {APIKey: "k"},
			},
		},
	}
	if _, _, err := resolveBenchmarkProvider(cfg, "custom", ""); err == nil || !strings.Contains(err.Error(), "unsupported provider") {
		t.Fatalf("expected unsupported provider error, got %v", err)
	}

	cfg = &config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: "claude",
			Providers: map[string]config.ProviderConfig{
				"claude": {APIKey: "k"},
				"openai": {APIKey: "k", Model: "gpt-4o-mini"},
			},
		},
	}
	if p, model, err := resolveBenchmarkProvider(cfg, "", ""); err != nil || p == nil || model != "default" {
		t.Fatalf("claude: p=%v model=%q err=%v", p, model, err)
	}
	if p, model, err := resolveBenchmarkProvider(cfg, "openai", "gpt-4o"); err != nil || p == nil || model != "gpt-4o" {
		t.Fatalf("openai: p=%v model=%q err=%v", p, model, err)
	}
}

func TestOpenLeaderboardStore(t *testing.T) {
	t.Parallel()

	if _, err := openLeaderboardStore(nil); err == nil {
		t.Fatalf("expected error for nil config")
	}

	cfg := &config.Config{Storage: config.StorageConfig{Type: "memory"}}
	st, err := openLeaderboardStore(cfg)
	if err != nil {
		t.Fatalf("openLeaderboardStore(memory): %v", err)
	}
	_ = st.Close()

	cfg = &config.Config{Storage: config.StorageConfig{Type: "nope"}}
	if _, err := openLeaderboardStore(cfg); err == nil {
		t.Fatalf("expected error for unsupported storage type")
	}
}
