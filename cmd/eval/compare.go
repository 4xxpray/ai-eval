package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/spf13/cobra"
)

var errRegression = errors.New("ai-eval: regression detected")

type compareOptions struct {
	promptName string
	v1         string
	v2         string
	trials     int
	output     string
}

func newCompareCmd(st *cliState) *cobra.Command {
	var opts compareOptions

	cmd := &cobra.Command{
		Use:   "compare",
		Short: "Compare two prompt versions against the same test suites",
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
			return runCompare(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.promptName, "prompt", "", "prompt name to compare")
	cmd.Flags().StringVar(&opts.v1, "v1", "", "version 1")
	cmd.Flags().StringVar(&opts.v2, "v2", "", "version 2")
	cmd.Flags().IntVar(&opts.trials, "trials", -1, "number of trials per case (overrides config)")
	cmd.Flags().StringVar(&opts.output, "output", "", "output format: table|json|github")

	_ = cmd.MarkFlagRequired("prompt")
	_ = cmd.MarkFlagRequired("v1")
	_ = cmd.MarkFlagRequired("v2")

	return cmd
}

func runCompare(cmd *cobra.Command, st *cliState, opts *compareOptions) error {
	if st == nil {
		return fmt.Errorf("compare: nil state")
	}
	if opts == nil {
		return fmt.Errorf("compare: nil options")
	}
	if st.cfg == nil {
		return fmt.Errorf("compare: missing config (internal error)")
	}

	promptName := strings.TrimSpace(opts.promptName)
	if promptName == "" {
		return fmt.Errorf("compare: missing --prompt")
	}
	v1Version := strings.TrimSpace(opts.v1)
	v2Version := strings.TrimSpace(opts.v2)
	if v1Version == "" || v2Version == "" {
		return fmt.Errorf("compare: missing --v1/--v2")
	}
	if v1Version == v2Version {
		return fmt.Errorf("compare: --v1 and --v2 must differ")
	}

	// Default to table for human-readable side-by-side diff.
	output, err := resolveOutputFormat(opts.output, "", false)
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}

	trials := st.cfg.Evaluation.Trials
	if opts.trials >= 0 {
		trials = opts.trials
	}
	if trials <= 0 {
		return fmt.Errorf("compare: trials must be > 0 (got %d)", trials)
	}

	threshold := st.cfg.Evaluation.Threshold
	if threshold < 0 || threshold > 1 {
		return fmt.Errorf("compare: threshold must be between 0 and 1 (got %v)", threshold)
	}

	concurrency := st.cfg.Evaluation.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	prompts, err := app.LoadPromptsRecursive(defaultPromptsDir)
	if err != nil {
		return err
	}

	p1, err := app.FindPromptByNameVersion(prompts, promptName, v1Version)
	if err != nil {
		return err
	}
	p2, err := app.FindPromptByNameVersion(prompts, promptName, v2Version)
	if err != nil {
		return err
	}

	suites, err := app.LoadTestSuites(defaultTestsDir)
	if err != nil {
		return err
	}
	suites = app.FilterSuitesByPrompt(suites, promptName)
	if len(suites) == 0 {
		return fmt.Errorf("compare: no test suites found for prompt %q", promptName)
	}
	sort.Slice(suites, func(i, j int) bool { return strings.ToLower(suites[i].Suite) < strings.ToLower(suites[j].Suite) })

	provider, err := llm.DefaultProviderFromConfig(st.cfg)
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})
	reg.Register(evaluator.ContainsEvaluator{})
	reg.Register(evaluator.NotContainsEvaluator{})
	reg.Register(evaluator.RegexEvaluator{})
	reg.Register(evaluator.JSONSchemaEvaluator{})

	r := runner.NewRunner(provider, reg, runner.Config{
		Trials:        trials,
		PassThreshold: threshold,
		Concurrency:   concurrency,
		Timeout:       st.cfg.Evaluation.Timeout,
	})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	out := cmd.OutOrStdout()
	regressed := false

	switch output {
	case FormatTable:
		_, _ = fmt.Fprintf(out, "Prompt: %s v1=%s v2=%s\n\n", promptName, v1Version, v2Version)
	case FormatGitHub:
		_, _ = fmt.Fprintf(out, "Summary: compare prompt=%s v1=%s v2=%s\n", promptName, v1Version, v2Version)
	case FormatJSON:
		meta := map[string]any{
			"compare": map[string]any{
				"prompt": promptName,
				"v1":     v1Version,
				"v2":     v2Version,
			},
		}
		if b, err := json.Marshal(meta); err == nil {
			_, _ = fmt.Fprintln(out, string(b))
		}
	}

	for _, suite := range suites {
		res1, err := r.RunSuite(ctx, p1, suite)
		if err != nil {
			return err
		}
		res2, err := r.RunSuite(ctx, p2, suite)
		if err != nil {
			return err
		}

		_, diffs := buildCompare(res1, res2)
		for _, d := range diffs {
			if d.Regression {
				regressed = true
				break
			}
		}

		_, _ = fmt.Fprint(out, FormatCompareResult(res1, res2, output))
	}

	if regressed {
		return errRegression
	}
	return nil
}
