package llm

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/config"
)

func NewRegistryFromConfig(cfg *config.Config) (*Registry, error) {
	if cfg == nil {
		return nil, errors.New("llm: nil config")
	}

	r := NewRegistry()
	for name, pcfg := range cfg.LLM.Providers {
		key := strings.ToLower(strings.TrimSpace(name))
		switch key {
		case "":
			continue
		case "claude", "anthropic":
			r.Register(NewClaudeProvider(pcfg.APIKey, pcfg.BaseURL, pcfg.Model))
		case "openai":
			r.Register(NewOpenAIProvider(pcfg.APIKey, pcfg.BaseURL, pcfg.Model))
		default:
			return nil, fmt.Errorf("llm: unknown provider %q", name)
		}
	}

	return r, nil
}

func DefaultProviderFromConfig(cfg *config.Config) (Provider, error) {
	if cfg == nil {
		return nil, errors.New("llm: nil config")
	}
	reg, err := NewRegistryFromConfig(cfg)
	if err != nil {
		return nil, err
	}

	name := strings.TrimSpace(cfg.LLM.DefaultProvider)
	if name == "" {
		name = "claude"
	}
	if p, ok := reg.Get(name); ok {
		return p, nil
	}

	if reg != nil && len(reg.providers) == 1 {
		for _, p := range reg.providers {
			return p, nil
		}
	}

	available := make([]string, 0, len(reg.providers))
	for k := range reg.providers {
		available = append(available, k)
	}
	sort.Strings(available)
	return nil, fmt.Errorf("llm: default provider %q not configured (available: %s)", name, strings.Join(available, ", "))
}
