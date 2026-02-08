package main

import (
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

var defaultProviderFromConfig = llm.DefaultProviderFromConfig

var benchmarkProviderFromConfig = func(cfg *config.Config, providerFlag, modelFlag string) (llm.Provider, string, error) {
	return resolveBenchmarkProvider(cfg, providerFlag, modelFlag)
}
