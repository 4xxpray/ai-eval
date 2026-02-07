package config

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const DefaultPath = "configs/config.yaml"

type Config struct {
	LLM        LLMConfig        `yaml:"llm"`
	Evaluation EvaluationConfig `yaml:"evaluation"`
	Storage    StorageConfig    `yaml:"storage"`
}

type LLMConfig struct {
	DefaultProvider string                    `yaml:"default_provider,omitempty"`
	Providers       map[string]ProviderConfig `yaml:"providers,omitempty"`
}

type ProviderConfig struct {
	APIKey  string `yaml:"api_key"`
	BaseURL string `yaml:"base_url,omitempty"`
	Model   string `yaml:"model,omitempty"`
}

type EvaluationConfig struct {
	Trials       int           `yaml:"trials"`
	Threshold    float64       `yaml:"threshold"`
	OutputFormat string        `yaml:"output_format,omitempty"`
	Concurrency  int           `yaml:"concurrency,omitempty"`
	Timeout      time.Duration `yaml:"timeout,omitempty"`
}

type StorageConfig struct {
	Type string `yaml:"type,omitempty"` // "sqlite" or "memory"
	Path string `yaml:"path,omitempty"` // SQLite file path
}

func Load(path string) (*Config, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		path = DefaultPath
	}

	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config: read %q: %w", path, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("config: parse %q: %w", path, err)
	}

	if cfg.LLM.Providers == nil {
		cfg.LLM.Providers = make(map[string]ProviderConfig)
	}

	if strings.TrimSpace(cfg.LLM.DefaultProvider) == "" {
		cfg.LLM.DefaultProvider = "claude"
	}

	if v := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); v != "" {
		p := cfg.LLM.Providers["claude"]
		p.APIKey = v
		cfg.LLM.Providers["claude"] = p
	} else if v := strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN")); v != "" {
		p := cfg.LLM.Providers["claude"]
		p.APIKey = v
		cfg.LLM.Providers["claude"] = p
	}

	if v := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); v != "" {
		p := cfg.LLM.Providers["openai"]
		p.APIKey = v
		cfg.LLM.Providers["openai"] = p
	}

	return &cfg, nil
}
