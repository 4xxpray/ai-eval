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
	"time"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

var errTestsFailed = errors.New("ai-eval: tests failed")

type runOptions struct {
	promptName string
	all        bool
	trials     int
	threshold  float64
	output     string
	ci         bool
}

func newRunCmd(st *cliState) *cobra.Command {
	var opts runOptions

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run evaluations",
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
			return runEvaluations(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.promptName, "prompt", "", "prompt name to run")
	cmd.Flags().BoolVar(&opts.all, "all", false, "run all prompts")
	cmd.Flags().IntVar(&opts.trials, "trials", -1, "number of trials per case (overrides config)")
	cmd.Flags().Float64Var(&opts.threshold, "threshold", -1, "pass@k threshold between 0 and 1 (overrides config)")
	cmd.Flags().StringVar(&opts.output, "output", "", "output format: table|json|github (overrides config)")
	cmd.Flags().BoolVar(&opts.ci, "ci", false, "force CI mode (github output and summaries)")

	return cmd
}

func runEvaluations(cmd *cobra.Command, st *cliState, opts *runOptions) error {
	if st == nil {
		return fmt.Errorf("run: nil state")
	}
	if opts == nil {
		return fmt.Errorf("run: nil options")
	}
	if st.cfg == nil {
		return fmt.Errorf("run: missing config (internal error)")
	}

	ciMode := resolveCIMode(opts)
	applyCIOutputDefaults(opts, ciMode)

	promptName := strings.TrimSpace(opts.promptName)
	switch {
	case opts.all && promptName != "":
		return fmt.Errorf("run: --all and --prompt are mutually exclusive")
	case !opts.all && promptName == "":
		return fmt.Errorf("run: specify either --prompt <name> or --all")
	}

	output, err := resolveOutputFormat(opts.output, st.cfg.Evaluation.OutputFormat, opts.all)
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}

	trials := st.cfg.Evaluation.Trials
	if opts.trials >= 0 {
		trials = opts.trials
	}
	if trials <= 0 {
		return fmt.Errorf("run: trials must be > 0 (got %d)", trials)
	}

	threshold := st.cfg.Evaluation.Threshold
	if opts.threshold >= 0 {
		threshold = opts.threshold
	}
	if threshold < 0 || threshold > 1 {
		return fmt.Errorf("run: threshold must be between 0 and 1 (got %v)", threshold)
	}

	concurrency := st.cfg.Evaluation.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	prompts, err := app.LoadPrompts(defaultPromptsDir)
	if err != nil {
		return err
	}
	promptByName, err := app.IndexPrompts(prompts)
	if err != nil {
		return err
	}

	suites, err := app.LoadTestSuites(defaultTestsDir)
	if err != nil {
		return err
	}
	suitesByPrompt, err := app.IndexSuitesByPrompt(suites, promptByName)
	if err != nil {
		return err
	}

	var promptNames []string
	if opts.all {
		for name := range suitesByPrompt {
			promptNames = append(promptNames, name)
		}
		sort.Strings(promptNames)
	} else {
		if _, ok := promptByName[promptName]; !ok {
			return fmt.Errorf("run: unknown prompt %q", promptName)
		}
		promptNames = []string{promptName}
	}
	if len(promptNames) == 0 {
		return fmt.Errorf("run: no test suites found")
	}

	provider, err := defaultProviderFromConfig(st.cfg)
	if err != nil {
		return fmt.Errorf("run: %w", err)
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

	startedAt := time.Now().UTC()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var runs []app.SuiteRun
	for _, name := range promptNames {
		p := promptByName[name]
		suites := suitesByPrompt[name]
		if len(suites) == 0 {
			return fmt.Errorf("run: no test suites found for prompt %q", name)
		}
		sort.Slice(suites, func(i, j int) bool { return suites[i].Suite < suites[j].Suite })

		for _, suite := range suites {
			res, _ := r.RunSuite(ctx, p, suite)
			runs = append(runs, app.SuiteRun{PromptName: name, PromptVersion: p.Version, Suite: suite, Result: res})
		}
	}

	finishedAt := time.Now().UTC()

	anyFailed, summary := app.SummarizeRuns(runs)
	switch output {
	case FormatTable:
		if !opts.all && len(promptNames) == 1 {
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Prompt: %s\n\n", promptNames[0])
		}
		for _, r := range runs {
			_, _ = fmt.Fprint(cmd.OutOrStdout(), FormatSuiteResult(r.Result, FormatTable))
		}

		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Summary: suites=%d cases=%d passed=%d failed=%d latency_ms=%d tokens=%d\n",
			summary.TotalSuites, summary.TotalCases, summary.PassedCases, summary.FailedCases, summary.TotalLatency, summary.TotalTokens)

		if summary.FailedCases == 0 {
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Overall: %s\n", coloredStatus(true))
		} else {
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Overall: %s\n", coloredStatus(false))
		}
	case FormatJSON:
		if err := printRunJSON(cmd, runs, summary); err != nil {
			return err
		}
	case FormatGitHub:
		for _, r := range runs {
			res := r.Result
			if res != nil && opts.all {
				tmp := *res
				tmp.Suite = fmt.Sprintf("%s (prompt=%s)", strings.TrimSpace(tmp.Suite), r.PromptName)
				res = &tmp
			}
			_, _ = fmt.Fprint(cmd.OutOrStdout(), FormatSuiteResult(res, FormatGitHub))
		}
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Summary: suites=%d cases=%d passed=%d failed=%d latency_ms=%d tokens=%d\n",
			summary.TotalSuites, summary.TotalCases, summary.PassedCases, summary.FailedCases, summary.TotalLatency, summary.TotalTokens)
	}

	if err := saveRunToStore(cmd.Context(), st, runs, summary, startedAt, finishedAt, promptNames, opts.all, output, trials, threshold, concurrency); err != nil {
		return err
	}

	if ciMode {
		writeCIArtifacts(runs, summary, startedAt, finishedAt, threshold)
	}

	if anyFailed {
		return errTestsFailed
	}
	return nil
}

type jsonRunSuiteLine struct {
	Prompt string           `json:"prompt"`
	Result *jsonSuiteResult `json:"result,omitempty"`
	Error  string           `json:"error,omitempty"`
	Suite  string           `json:"suite,omitempty"`
}

type jsonRunSummaryLine struct {
	Summary app.RunSummary `json:"summary"`
}

func printRunJSON(cmd *cobra.Command, runs []app.SuiteRun, summary app.RunSummary) error {
	out := cmd.OutOrStdout()
	enc := json.NewEncoder(out)

	for _, r := range runs {
		line := jsonRunSuiteLine{
			Prompt: r.PromptName,
		}
		if r.Result == nil {
			line.Error = "nil suite result"
			if r.Suite != nil {
				line.Suite = r.Suite.Suite
			}
		} else {
			tmp := suiteResultToJSON(r.Result)
			line.Result = &tmp
		}

		if err := enc.Encode(line); err != nil {
			return fmt.Errorf("run: marshal json: %w", err)
		}
	}

	sumLine := jsonRunSummaryLine{
		Summary: summary,
	}
	if err := enc.Encode(sumLine); err != nil {
		return fmt.Errorf("run: marshal json: %w", err)
	}
	return nil
}

func saveRunToStore(ctx context.Context, st *cliState, runs []app.SuiteRun, summary app.RunSummary, startedAt, finishedAt time.Time, promptNames []string, all bool, output OutputFormat, trials int, threshold float64, concurrency int) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("run: missing config (internal error)")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	stor, err := store.Open(st.cfg)
	if err != nil {
		return fmt.Errorf("run: open store: %w", err)
	}
	defer stor.Close()

	_, err = app.SaveRun(ctx, stor, runs, summary, startedAt, finishedAt, buildRunConfig(st, promptNames, all, output, trials, threshold, concurrency))
	return err
}

func buildRunConfig(st *cliState, promptNames []string, all bool, output OutputFormat, trials int, threshold float64, concurrency int) map[string]any {
	cfg := map[string]any{
		"output":      string(output),
		"trials":      trials,
		"threshold":   threshold,
		"concurrency": concurrency,
		"all":         all,
	}
	if len(promptNames) > 0 {
		cfg["prompts"] = append([]string(nil), promptNames...)
	}
	if st != nil && st.cfg != nil && st.cfg.Evaluation.Timeout > 0 {
		cfg["timeout_ms"] = st.cfg.Evaluation.Timeout.Milliseconds()
	}
	return cfg
}
