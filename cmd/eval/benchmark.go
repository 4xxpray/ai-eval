package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/benchmark"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/spf13/cobra"
)

type benchmarkOptions struct {
	model      string
	provider   string
	dataset    string
	sampleSize int
}

func newBenchmarkCmd(st *cliState) *cobra.Command {
	var opts benchmarkOptions

	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Run a benchmark dataset and save results",
		Args:  cobra.NoArgs,
		PreRunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(st.configPath)
			if err != nil {
				return err
			}
			st.cfg = cfg
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			return runBenchmark(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.model, "model", "", "model name (overrides config)")
	cmd.Flags().StringVar(&opts.provider, "provider", "", "provider name (overrides config)")
	cmd.Flags().StringVar(&opts.dataset, "dataset", "", "dataset: mmlu|humaneval|gsm8k")
	cmd.Flags().IntVar(&opts.sampleSize, "sample-size", 0, "sample size (0 = default)")

	return cmd
}

func runBenchmark(cmd *cobra.Command, st *cliState, opts *benchmarkOptions) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("benchmark: missing config (internal error)")
	}
	if opts == nil {
		return fmt.Errorf("benchmark: nil options")
	}

	ds, err := resolveBenchmarkDataset(opts.dataset, opts.sampleSize)
	if err != nil {
		return err
	}

	provider, modelName, err := resolveBenchmarkProvider(st.cfg, opts.provider, opts.model)
	if err != nil {
		return err
	}

	lb, err := openLeaderboardStore(st.cfg)
	if err != nil {
		return err
	}
	defer lb.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	r := &benchmark.BenchmarkRunner{
		Provider: provider,
		Store:    lb,
	}
	res, runErr := r.Run(ctx, ds)
	if res == nil {
		return runErr
	}
	if runErr != nil {
		return runErr
	}
	res.Model = modelName

	entry := &leaderboard.Entry{
		Model:    modelName,
		Provider: provider.Name(),
		Dataset:  ds.Name(),
		Score:    res.Score,
		Accuracy: res.Accuracy,
		Latency:  res.TotalTime.Milliseconds(),
		Cost:     0,
		EvalDate: time.Now().UTC(),
	}
	if err := lb.Save(cmd.Context(), entry); err != nil {
		return err
	}

	out := cmd.OutOrStdout()
	_, _ = fmt.Fprintf(out, "Benchmark saved: id=%d provider=%s model=%s dataset=%s score=%.4f accuracy=%.4f time_ms=%d tokens=%d\n",
		entry.ID,
		entry.Provider,
		entry.Model,
		entry.Dataset,
		entry.Score,
		entry.Accuracy,
		entry.Latency,
		res.TotalTokens,
	)

	return nil
}

func resolveBenchmarkDataset(name string, sampleSize int) (benchmark.Dataset, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return nil, fmt.Errorf("benchmark: missing --dataset (mmlu|humaneval|gsm8k)")
	}
	if sampleSize < 0 {
		return nil, fmt.Errorf("benchmark: --sample-size must be >= 0 (got %d)", sampleSize)
	}

	switch name {
	case "mmlu":
		return &benchmark.MMLUDataset{SampleSize: sampleSize}, nil
	case "humaneval":
		return &benchmark.HumanEvalDataset{SampleSize: sampleSize}, nil
	case "gsm8k":
		return &benchmark.GSM8KDataset{SampleSize: sampleSize}, nil
	default:
		return nil, fmt.Errorf("benchmark: unknown dataset %q (expected mmlu|humaneval|gsm8k)", name)
	}
}

func resolveBenchmarkProvider(cfg *config.Config, providerFlag string, modelFlag string) (llm.Provider, string, error) {
	if cfg == nil {
		return nil, "", fmt.Errorf("benchmark: missing config")
	}

	providerName := strings.TrimSpace(providerFlag)
	if providerName == "" {
		providerName = strings.TrimSpace(cfg.LLM.DefaultProvider)
	}
	providerName = normalizeProvider(providerName)
	if providerName == "" {
		return nil, "", fmt.Errorf("benchmark: missing provider")
	}

	pcfg, ok := cfg.LLM.Providers[providerName]
	if !ok {
		available := make([]string, 0, len(cfg.LLM.Providers))
		for k := range cfg.LLM.Providers {
			available = append(available, k)
		}
		sort.Strings(available)
		return nil, "", fmt.Errorf("benchmark: provider %q not configured (available: %s)", providerName, strings.Join(available, ", "))
	}

	model := strings.TrimSpace(modelFlag)
	if model == "" {
		model = strings.TrimSpace(pcfg.Model)
	}
	modelName := model
	if modelName == "" {
		modelName = "default"
	}

	switch providerName {
	case "claude":
		return llm.NewClaudeProvider(pcfg.APIKey, pcfg.BaseURL, model), modelName, nil
	case "openai":
		return llm.NewOpenAIProvider(pcfg.APIKey, pcfg.BaseURL, model), modelName, nil
	default:
		return nil, "", fmt.Errorf("benchmark: unsupported provider %q", providerName)
	}
}

func normalizeProvider(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	switch name {
	case "anthropic":
		return "claude"
	default:
		return name
	}
}

func openLeaderboardStore(cfg *config.Config) (*leaderboard.Store, error) {
	if cfg == nil {
		return nil, fmt.Errorf("leaderboard: missing config")
	}

	storageType := strings.ToLower(strings.TrimSpace(cfg.Storage.Type))
	if storageType == "" {
		storageType = "sqlite"
	}

	switch storageType {
	case "sqlite":
		path := strings.TrimSpace(cfg.Storage.Path)
		if path == "" {
			path = store.DefaultSQLitePath
		}
		return leaderboard.NewStore(path)
	case "memory":
		return leaderboard.NewStore(":memory:")
	default:
		return nil, fmt.Errorf("leaderboard: unsupported type %q", storageType)
	}
}
