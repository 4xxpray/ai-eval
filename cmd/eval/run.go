package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/app"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
	"github.com/spf13/cobra"
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

type suiteRun struct {
	promptName    string
	promptVersion string
	suite         *testcase.TestSuite
	result        *runner.SuiteResult
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
	promptByName, err := indexPrompts(prompts)
	if err != nil {
		return err
	}

	suites, err := app.LoadTestSuites(defaultTestsDir)
	if err != nil {
		return err
	}
	suitesByPrompt, err := indexSuitesByPrompt(suites, promptByName)
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

	provider, err := llm.DefaultProviderFromConfig(st.cfg)
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

	var runs []suiteRun
	for _, name := range promptNames {
		p := promptByName[name]
		suites := suitesByPrompt[name]
		if len(suites) == 0 {
			return fmt.Errorf("run: no test suites found for prompt %q", name)
		}
		sort.Slice(suites, func(i, j int) bool { return suites[i].Suite < suites[j].Suite })

		for _, suite := range suites {
			res, err := r.RunSuite(ctx, p, suite)
			if err != nil {
				return err
			}
			runs = append(runs, suiteRun{promptName: name, promptVersion: p.Version, suite: suite, result: res})
		}
	}

	finishedAt := time.Now().UTC()

	anyFailed, summary := summarizeRuns(runs)
	switch output {
	case FormatTable:
		if !opts.all && len(promptNames) == 1 {
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Prompt: %s\n\n", promptNames[0])
		}
		for _, r := range runs {
			_, _ = fmt.Fprint(cmd.OutOrStdout(), FormatSuiteResult(r.result, FormatTable))
		}

		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Summary: suites=%d cases=%d passed=%d failed=%d latency_ms=%d tokens=%d\n",
			summary.totalSuites, summary.totalCases, summary.passedCases, summary.failedCases, summary.totalLatency, summary.totalTokens)

		if summary.failedCases == 0 {
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
			res := r.result
			if res != nil && opts.all {
				tmp := *res
				tmp.Suite = fmt.Sprintf("%s (prompt=%s)", strings.TrimSpace(tmp.Suite), r.promptName)
				res = &tmp
			}
			_, _ = fmt.Fprint(cmd.OutOrStdout(), FormatSuiteResult(res, FormatGitHub))
		}
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Summary: suites=%d cases=%d passed=%d failed=%d latency_ms=%d tokens=%d\n",
			summary.totalSuites, summary.totalCases, summary.passedCases, summary.failedCases, summary.totalLatency, summary.totalTokens)
	default:
		return fmt.Errorf("run: internal error: unknown output format %q", output)
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

func indexPrompts(prompts []*prompt.Prompt) (map[string]*prompt.Prompt, error) {
	out := make(map[string]*prompt.Prompt, len(prompts))
	for _, p := range prompts {
		if p == nil {
			return nil, fmt.Errorf("run: nil prompt")
		}
		name := strings.TrimSpace(p.Name)
		if name == "" {
			return nil, fmt.Errorf("run: prompt with empty name")
		}
		if _, ok := out[name]; ok {
			return nil, fmt.Errorf("run: duplicate prompt name %q", name)
		}
		out[name] = p
	}
	return out, nil
}

func indexSuitesByPrompt(suites []*testcase.TestSuite, promptByName map[string]*prompt.Prompt) (map[string][]*testcase.TestSuite, error) {
	out := make(map[string][]*testcase.TestSuite)
	for _, s := range suites {
		if s == nil {
			return nil, fmt.Errorf("run: nil test suite")
		}
		promptRef := strings.TrimSpace(s.Prompt)
		if promptRef == "" {
			return nil, fmt.Errorf("run: suite %q: missing prompt reference", s.Suite)
		}
		if _, ok := promptByName[promptRef]; !ok {
			return nil, fmt.Errorf("run: suite %q references unknown prompt %q", s.Suite, promptRef)
		}
		out[promptRef] = append(out[promptRef], s)
	}
	return out, nil
}

type runSummary struct {
	totalSuites  int
	totalCases   int
	passedCases  int
	failedCases  int
	totalLatency int64
	totalTokens  int
}

func summarizeRuns(runs []suiteRun) (anyFailed bool, summary runSummary) {
	summary.totalSuites = len(runs)
	for _, r := range runs {
		if r.result == nil {
			anyFailed = true
			continue
		}
		summary.totalCases += r.result.TotalCases
		summary.passedCases += r.result.PassedCases
		summary.failedCases += r.result.FailedCases
		summary.totalLatency += r.result.TotalLatency
		summary.totalTokens += r.result.TotalTokens
		if r.result.FailedCases > 0 {
			anyFailed = true
		}
	}
	if summary.failedCases > 0 {
		anyFailed = true
	}
	return anyFailed, summary
}

type jsonRunSuiteLine struct {
	Prompt string           `json:"prompt"`
	Result *jsonSuiteResult `json:"result,omitempty"`
	Error  string           `json:"error,omitempty"`
	Suite  string           `json:"suite,omitempty"`
}

type jsonRunSummaryLine struct {
	Summary jsonRunSummary `json:"summary"`
}

type jsonRunSummary struct {
	TotalSuites  int   `json:"total_suites"`
	TotalCases   int   `json:"total_cases"`
	PassedCases  int   `json:"passed_cases"`
	FailedCases  int   `json:"failed_cases"`
	TotalLatency int64 `json:"total_latency_ms"`
	TotalTokens  int   `json:"total_tokens"`
}

func printRunJSON(cmd *cobra.Command, runs []suiteRun, summary runSummary) error {
	out := cmd.OutOrStdout()

	for _, r := range runs {
		line := jsonRunSuiteLine{
			Prompt: r.promptName,
		}
		if r.result == nil {
			line.Error = "nil suite result"
			if r.suite != nil {
				line.Suite = r.suite.Suite
			}
		} else {
			tmp := suiteResultToJSON(r.result)
			line.Result = &tmp
		}

		b, err := json.Marshal(line)
		if err != nil {
			return fmt.Errorf("run: marshal json: %w", err)
		}
		_, _ = fmt.Fprintln(out, string(b))
	}

	sumLine := jsonRunSummaryLine{
		Summary: jsonRunSummary{
			TotalSuites:  summary.totalSuites,
			TotalCases:   summary.totalCases,
			PassedCases:  summary.passedCases,
			FailedCases:  summary.failedCases,
			TotalLatency: summary.totalLatency,
			TotalTokens:  summary.totalTokens,
		},
	}
	b, err := json.Marshal(sumLine)
	if err != nil {
		return fmt.Errorf("run: marshal json: %w", err)
	}
	_, _ = fmt.Fprintln(out, string(b))
	return nil
}

func saveRunToStore(ctx context.Context, st *cliState, runs []suiteRun, summary runSummary, startedAt, finishedAt time.Time, promptNames []string, all bool, output OutputFormat, trials int, threshold float64, concurrency int) error {
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

	var writer store.RunWriter = stor

	runID, err := newRunID()
	if err != nil {
		return fmt.Errorf("run: generate run id: %w", err)
	}

	passedSuites := 0
	failedSuites := 0
	for _, r := range runs {
		if r.result != nil && r.result.FailedCases == 0 {
			passedSuites++
		} else {
			failedSuites++
		}
	}

	runRecord := &store.RunRecord{
		ID:           runID,
		StartedAt:    startedAt,
		FinishedAt:   finishedAt,
		TotalSuites:  summary.totalSuites,
		PassedSuites: passedSuites,
		FailedSuites: failedSuites,
		Config:       buildRunConfig(st, promptNames, all, output, trials, threshold, concurrency),
	}
	if err := writer.SaveRun(ctx, runRecord); err != nil {
		return fmt.Errorf("run: save run: %w", err)
	}

	for i, r := range runs {
		if r.result == nil || r.suite == nil {
			return fmt.Errorf("run: missing suite result")
		}
		caseResults := make([]store.CaseRecord, 0, len(r.result.Results))
		for _, rr := range r.result.Results {
			cr := store.CaseRecord{
				CaseID:     rr.CaseID,
				Passed:     rr.Passed,
				Score:      rr.Score,
				PassAtK:    rr.PassAtK,
				PassExpK:   rr.PassExpK,
				LatencyMs:  rr.LatencyMs,
				TokensUsed: rr.TokensUsed,
			}
			if rr.Error != nil {
				cr.Error = rr.Error.Error()
			}
			caseResults = append(caseResults, cr)
		}

		suiteRecord := &store.SuiteRecord{
			ID:            fmt.Sprintf("%s_suite_%d", runID, i+1),
			RunID:         runID,
			PromptName:    r.promptName,
			PromptVersion: r.promptVersion,
			SuiteName:     r.suite.Suite,
			TotalCases:    r.result.TotalCases,
			PassedCases:   r.result.PassedCases,
			FailedCases:   r.result.FailedCases,
			PassRate:      r.result.PassRate,
			AvgScore:      r.result.AvgScore,
			TotalLatency:  r.result.TotalLatency,
			TotalTokens:   r.result.TotalTokens,
			CreatedAt:     finishedAt,
			CaseResults:   caseResults,
		}
		if err := writer.SaveSuiteResult(ctx, suiteRecord); err != nil {
			return fmt.Errorf("run: save suite result: %w", err)
		}
	}

	return nil
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

func newRunID() (string, error) {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("run_%s_%x", time.Now().UTC().Format("20060102T150405Z"), buf), nil
}
