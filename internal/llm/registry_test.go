package llm

import (
	"context"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
)

type stubProvider struct {
	name string
}

func (p stubProvider) Name() string { return p.name }
func (p stubProvider) Complete(context.Context, *Request) (*Response, error) {
	return nil, nil
}
func (p stubProvider) CompleteWithTools(context.Context, *Request) (*EvalResult, error) {
	return nil, nil
}

func TestRegistry_RegisterAndGet(t *testing.T) {
	t.Parallel()

	var nilReg *Registry
	nilReg.Register(stubProvider{name: "x"}) // should be no-op

	r := &Registry{}
	r.Register(stubProvider{name: " \t "}) // should be ignored
	if _, ok := r.Get("x"); ok {
		t.Fatalf("Get: unexpected provider")
	}

	r.Register(nil)
	r.Register(stubProvider{name: "  X "})

	if r.providers == nil {
		t.Fatalf("providers: nil")
	}
	if got, ok := r.Get("x"); !ok || got == nil {
		t.Fatalf("Get(x): ok=%v provider=%v", ok, got)
	}
	if _, ok := r.Get(" \t "); ok {
		t.Fatalf("Get(empty): unexpected ok")
	}
}

func TestNewRegistryFromConfig(t *testing.T) {
	t.Parallel()

	if _, err := NewRegistryFromConfig(nil); err == nil {
		t.Fatalf("NewRegistryFromConfig(nil): expected error")
	}

	_, err := NewRegistryFromConfig(&config.Config{
		LLM: config.LLMConfig{
			Providers: map[string]config.ProviderConfig{
				"unknown": {},
			},
		},
	})
	if err == nil {
		t.Fatalf("NewRegistryFromConfig: expected error")
	}
	if !strings.Contains(err.Error(), "unknown provider") {
		t.Fatalf("error: got %q", err.Error())
	}

	reg, err := NewRegistryFromConfig(&config.Config{
		LLM: config.LLMConfig{
			Providers: map[string]config.ProviderConfig{
				"  ":        {},
				"OpenAI":    {APIKey: "k1", BaseURL: "http://example.test/v1", Model: "gpt-4o"},
				"Anthropic": {APIKey: "k2"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewRegistryFromConfig: %v", err)
	}
	if _, ok := reg.Get("openai"); !ok {
		t.Fatalf("Get(openai): not found")
	}
	if _, ok := reg.Get("claude"); !ok {
		t.Fatalf("Get(claude): not found")
	}
}

func TestDefaultProviderFromConfig(t *testing.T) {
	t.Parallel()

	if _, err := DefaultProviderFromConfig(nil); err == nil {
		t.Fatalf("DefaultProviderFromConfig(nil): expected error")
	}

	p, err := DefaultProviderFromConfig(&config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: " \t ",
			Providers: map[string]config.ProviderConfig{
				"openai": {APIKey: "k"},
			},
		},
	})
	if err != nil {
		t.Fatalf("DefaultProviderFromConfig(single provider): %v", err)
	}
	if p == nil || p.Name() != "openai" {
		t.Fatalf("provider: got %#v", p)
	}

	p, err = DefaultProviderFromConfig(&config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: "openai",
			Providers: map[string]config.ProviderConfig{
				"openai": {APIKey: "k1"},
				"claude": {APIKey: "k2"},
			},
		},
	})
	if err != nil {
		t.Fatalf("DefaultProviderFromConfig(configured default): %v", err)
	}
	if p == nil || p.Name() != "openai" {
		t.Fatalf("provider: got %#v", p)
	}

	_, err = DefaultProviderFromConfig(&config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: "anthropic",
			Providers: map[string]config.ProviderConfig{
				"openai": {APIKey: "k1"},
				"claude": {APIKey: "k2"},
			},
		},
	})
	if err == nil {
		t.Fatalf("DefaultProviderFromConfig(bad default): expected error")
	}
	if !strings.Contains(err.Error(), "available: claude, openai") {
		t.Fatalf("error: got %q", err.Error())
	}

	_, err = DefaultProviderFromConfig(&config.Config{
		LLM: config.LLMConfig{
			DefaultProvider: "claude",
			Providers:       map[string]config.ProviderConfig{},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "available:") {
		t.Fatalf("DefaultProviderFromConfig(empty): got %v", err)
	}
}
