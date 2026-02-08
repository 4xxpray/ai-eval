package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

type diagnoseOptions struct {
	promptPath string
	testsPath  string
	output     string
}

func newDiagnoseCmd(st *cliState) *cobra.Command {
	var opts diagnoseOptions

	cmd := &cobra.Command{
		Use:   "diagnose",
		Short: "Diagnose evaluation failures and propose targeted prompt fixes",
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
			return runDiagnose(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.promptPath, "prompt", "", "path to prompt file (.yaml/.yml or plain text); if omitted, read from stdin")
	cmd.Flags().StringVar(&opts.testsPath, "tests", defaultTestsDir, "path to test suite file or directory")
	cmd.Flags().StringVar(&opts.output, "output", "text", "output format: text|json")

	return cmd
}

type promptInput struct {
	Path        string
	IsYAML      bool
	Prompt      *prompt.Prompt
	PromptText  string
	NameHint    string
	SystemHint  *bool
	SourceLabel string
}

func runDiagnose(cmd *cobra.Command, st *cliState, opts *diagnoseOptions) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("diagnose: missing config (internal error)")
	}
	if opts == nil {
		return fmt.Errorf("diagnose: nil options")
	}

	outFmt := strings.ToLower(strings.TrimSpace(opts.output))
	if outFmt == "" {
		outFmt = "text"
	}
	if outFmt != "text" && outFmt != "json" {
		return fmt.Errorf("diagnose: invalid --output %q (expected text|json)", opts.output)
	}

	provider, err := defaultProviderFromConfig(st.cfg)
	if err != nil {
		return fmt.Errorf("diagnose: %w", err)
	}

	r, err := newRunnerFromConfig(provider, st.cfg)
	if err != nil {
		return err
	}

	pIn, err := loadPromptInput(opts.promptPath)
	if err != nil {
		return err
	}

	suites, err := loadTestSuites(opts.testsPath)
	if err != nil {
		return err
	}

	suites, promptName, isSystem, err := selectSuitesAndPromptHints(pIn, suites)
	if err != nil {
		return err
	}

	p := buildPromptForRun(pIn, promptName, isSystem)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()

	results, _ := runSuites(ctx, r, p, suites)

	advisor := &optimizer.Advisor{Provider: provider}
	diag, err := advisor.Diagnose(ctx, &optimizer.DiagnoseRequest{
		PromptContent: pIn.PromptText,
		EvalResults:   results,
	})
	if err != nil {
		return err
	}

	switch outFmt {
	case "json":
		payload := buildDiagnoseJSONOutput(p, suites, results, diag)
		enc := json.NewEncoder(cmd.OutOrStdout())
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			return fmt.Errorf("diagnose: marshal output: %w", err)
		}
		return nil
	default:
		printDiagnoseText(cmd, p, suites, results, diag)
		return nil
	}
}

func newRunnerFromConfig(provider llm.Provider, cfg *config.Config) (*runner.Runner, error) {
	if provider == nil {
		return nil, fmt.Errorf("diagnose: nil llm provider")
	}
	if cfg == nil {
		return nil, fmt.Errorf("diagnose: nil config")
	}

	trials := cfg.Evaluation.Trials
	if trials <= 0 {
		trials = 1
	}

	threshold := cfg.Evaluation.Threshold
	if threshold < 0 || threshold > 1 {
		return nil, fmt.Errorf("diagnose: threshold must be between 0 and 1 (got %v)", threshold)
	}

	concurrency := cfg.Evaluation.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})
	reg.Register(evaluator.ContainsEvaluator{})
	reg.Register(evaluator.NotContainsEvaluator{})
	reg.Register(evaluator.RegexEvaluator{})
	reg.Register(evaluator.JSONSchemaEvaluator{})

	return runner.NewRunner(provider, reg, runner.Config{
		Trials:        trials,
		PassThreshold: threshold,
		Concurrency:   concurrency,
		Timeout:       cfg.Evaluation.Timeout,
	}), nil
}

func loadPromptInput(path string) (*promptInput, error) {
	path = strings.TrimSpace(path)
	if path != "" {
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".yaml" || ext == ".yml" {
			p, err := prompt.LoadFromFile(path)
			if err != nil {
				return nil, fmt.Errorf("diagnose: load prompt %q: %w", path, err)
			}
			text := p.Template
			if strings.TrimSpace(text) == "" {
				return nil, fmt.Errorf("diagnose: prompt %q has empty template", path)
			}
			nameHint := strings.TrimSpace(p.Name)
			if nameHint == "" {
				nameHint = strings.TrimSuffix(filepath.Base(path), ext)
			}
			system := p.IsSystemPrompt
			return &promptInput{
				Path:        path,
				IsYAML:      true,
				Prompt:      p,
				PromptText:  text,
				NameHint:    nameHint,
				SystemHint:  &system,
				SourceLabel: path,
			}, nil
		}

		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("diagnose: read prompt %q: %w", path, err)
		}
		text := string(b)
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("diagnose: prompt %q is empty", path)
		}
		nameHint := strings.TrimSuffix(filepath.Base(path), ext)
		return &promptInput{
			Path:        path,
			IsYAML:      false,
			PromptText:  text,
			NameHint:    nameHint,
			SourceLabel: path,
		}, nil
	}

	stat, _ := os.Stdin.Stat()
	if (stat.Mode() & os.ModeCharDevice) != 0 {
		return nil, errors.New("diagnose: no prompt provided: use --prompt or pipe content to stdin")
	}
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		return nil, fmt.Errorf("diagnose: read stdin: %w", err)
	}
	text := string(b)
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("diagnose: prompt content is empty")
	}
	return &promptInput{
		PromptText:  text,
		NameHint:    "prompt",
		SourceLabel: "stdin",
	}, nil
}

func loadTestSuites(path string) ([]*testcase.TestSuite, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		path = defaultTestsDir
	}

	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("diagnose: stat tests %q: %w", path, err)
	}
	if fi.IsDir() {
		suites, err := testcase.LoadFromDir(path)
		if err != nil {
			return nil, fmt.Errorf("diagnose: load tests dir %q: %w", path, err)
		}
		return compactSuites(suites), nil
	}

	s, err := testcase.LoadFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("diagnose: load tests file %q: %w", path, err)
	}
	return []*testcase.TestSuite{s}, nil
}

func selectSuitesAndPromptHints(pIn *promptInput, suites []*testcase.TestSuite) ([]*testcase.TestSuite, string, bool, error) {
	if pIn == nil {
		return nil, "", false, fmt.Errorf("diagnose: nil prompt input")
	}
	if len(suites) == 0 {
		return nil, "", false, fmt.Errorf("diagnose: no test suites loaded")
	}

	suites = compactSuites(suites)
	if len(suites) == 0 {
		return nil, "", false, fmt.Errorf("diagnose: no test suites loaded")
	}

	// Infer/validate prompt name from suites if prompt is not YAML-based.
	uniquePromptNames := make(map[string]struct{}, 4)
	for _, s := range suites {
		name := strings.TrimSpace(s.Prompt)
		if name == "" {
			return nil, "", false, fmt.Errorf("diagnose: suite %q: missing prompt reference", strings.TrimSpace(s.Suite))
		}
		uniquePromptNames[name] = struct{}{}
	}

	var suitePromptName string
	if len(uniquePromptNames) == 1 {
		for n := range uniquePromptNames {
			suitePromptName = n
		}
	}

	promptName := strings.TrimSpace(pIn.NameHint)
	if pIn.IsYAML && pIn.Prompt != nil {
		promptName = strings.TrimSpace(pIn.Prompt.Name)
	}

	if pIn.IsYAML {
		if suitePromptName == "" {
			return nil, "", false, fmt.Errorf("diagnose: tests contain multiple prompt names; expected exactly one")
		}
		if strings.TrimSpace(promptName) != "" && suitePromptName != strings.TrimSpace(promptName) {
			return nil, "", false, fmt.Errorf("diagnose: prompt name mismatch: prompt=%q tests=%q", promptName, suitePromptName)
		}
		promptName = suitePromptName
	} else {
		// For plain text prompts, only allow suites for one prompt to avoid accidental mismatches.
		if suitePromptName == "" {
			return nil, "", false, fmt.Errorf("diagnose: tests contain multiple prompt names; pass a specific suite file or use a YAML prompt")
		}
		promptName = suitePromptName
	}

	// Resolve system-prompt behavior.
	var suiteIsSystem *bool
	for _, s := range suites {
		v := s.IsSystemPrompt
		if suiteIsSystem == nil {
			suiteIsSystem = &v
			continue
		}
		if *suiteIsSystem != v {
			return nil, "", false, fmt.Errorf("diagnose: mixed is_system_prompt across suites for prompt %q", promptName)
		}
	}
	isSystem := false
	if suiteIsSystem != nil {
		isSystem = *suiteIsSystem
	}

	if pIn.SystemHint != nil && *pIn.SystemHint != isSystem {
		return nil, "", false, fmt.Errorf("diagnose: is_system_prompt mismatch: prompt=%v tests=%v", *pIn.SystemHint, isSystem)
	}

	sort.Slice(suites, func(i, j int) bool {
		return strings.ToLower(strings.TrimSpace(suites[i].Suite)) < strings.ToLower(strings.TrimSpace(suites[j].Suite))
	})

	return suites, promptName, isSystem, nil
}

func buildPromptForRun(pIn *promptInput, name string, isSystem bool) *prompt.Prompt {
	if pIn != nil && pIn.IsYAML && pIn.Prompt != nil {
		p := *pIn.Prompt
		p.Name = name
		p.IsSystemPrompt = isSystem
		return &p
	}

	return &prompt.Prompt{
		Name:           name,
		Template:       pIn.PromptText,
		IsSystemPrompt: isSystem,
	}
}

func compactSuites(in []*testcase.TestSuite) []*testcase.TestSuite {
	if len(in) == 0 {
		return in
	}
	out := in[:0]
	for _, s := range in {
		if s != nil {
			out = append(out, s)
		}
	}
	return out
}

func runSuites(ctx context.Context, r *runner.Runner, p *prompt.Prompt, suites []*testcase.TestSuite) ([]*runner.SuiteResult, error) {
	if r == nil {
		return nil, fmt.Errorf("diagnose: nil runner")
	}
	if p == nil {
		return nil, fmt.Errorf("diagnose: nil prompt")
	}

	out := make([]*runner.SuiteResult, 0, len(suites))
	for _, s := range suites {
		if s == nil {
			continue
		}
		res, err := r.RunSuite(ctx, p, s)
		if err != nil {
			return nil, err
		}
		out = append(out, res)
	}
	return out, nil
}

type diagnoseJSONOutput struct {
	PromptName string `json:"prompt_name"`
	Suites     []struct {
		Suite      string  `json:"suite"`
		PassRate   float64 `json:"pass_rate"`
		AvgScore   float64 `json:"avg_score"`
		TotalCases int     `json:"total_cases"`
		Passed     int     `json:"passed"`
		Failed     int     `json:"failed"`
	} `json:"suites"`
	Diagnosis *optimizer.DiagnoseResult `json:"diagnosis"`
}

func buildDiagnoseJSONOutput(p *prompt.Prompt, suites []*testcase.TestSuite, results []*runner.SuiteResult, diag *optimizer.DiagnoseResult) diagnoseJSONOutput {
	out := diagnoseJSONOutput{
		Diagnosis: diag,
	}
	if p != nil {
		out.PromptName = strings.TrimSpace(p.Name)
	}
	out.Suites = make([]struct {
		Suite      string  `json:"suite"`
		PassRate   float64 `json:"pass_rate"`
		AvgScore   float64 `json:"avg_score"`
		TotalCases int     `json:"total_cases"`
		Passed     int     `json:"passed"`
		Failed     int     `json:"failed"`
	}, 0, len(results))

	_ = suites
	for _, res := range results {
		if res == nil {
			continue
		}
		out.Suites = append(out.Suites, struct {
			Suite      string  `json:"suite"`
			PassRate   float64 `json:"pass_rate"`
			AvgScore   float64 `json:"avg_score"`
			TotalCases int     `json:"total_cases"`
			Passed     int     `json:"passed"`
			Failed     int     `json:"failed"`
		}{
			Suite:      strings.TrimSpace(res.Suite),
			PassRate:   res.PassRate,
			AvgScore:   res.AvgScore,
			TotalCases: res.TotalCases,
			Passed:     res.PassedCases,
			Failed:     res.FailedCases,
		})
	}

	return out
}

func printDiagnoseText(cmd *cobra.Command, p *prompt.Prompt, suites []*testcase.TestSuite, results []*runner.SuiteResult, diag *optimizer.DiagnoseResult) {
	out := cmd.OutOrStdout()

	promptName := ""
	if p != nil {
		promptName = strings.TrimSpace(p.Name)
	}
	if promptName != "" {
		_, _ = fmt.Fprintf(out, "Prompt: %s\n", promptName)
	}

	for _, res := range results {
		if res == nil {
			continue
		}
		_, _ = fmt.Fprintf(out, "\nSuite: %s\n", strings.TrimSpace(res.Suite))
		_, _ = fmt.Fprintf(out, "Cases: %d passed=%d failed=%d pass_rate=%.2f avg_score=%.2f\n",
			res.TotalCases, res.PassedCases, res.FailedCases, res.PassRate, res.AvgScore)
	}

	_ = suites

	if diag == nil {
		_, _ = fmt.Fprintln(out, "\nDiagnosis: <nil>")
		return
	}

	_, _ = fmt.Fprintln(out, "\nFailure Patterns:")
	if len(diag.FailurePatterns) == 0 {
		_, _ = fmt.Fprintln(out, "- (none)")
	} else {
		for _, p := range diag.FailurePatterns {
			_, _ = fmt.Fprintf(out, "- %s\n", p)
		}
	}

	_, _ = fmt.Fprintln(out, "\nRoot Causes:")
	if len(diag.RootCauses) == 0 {
		_, _ = fmt.Fprintln(out, "- (none)")
	} else {
		for _, rc := range diag.RootCauses {
			_, _ = fmt.Fprintf(out, "- %s\n", rc)
		}
	}

	_, _ = fmt.Fprintln(out, "\nSuggestions:")
	if len(diag.Suggestions) == 0 {
		_, _ = fmt.Fprintln(out, "- (none)")
		return
	}
	for _, s := range diag.Suggestions {
		_, _ = fmt.Fprintf(out, "- [%s] (priority=%d impact=%s type=%s) %s\n", s.ID, s.Priority, s.Impact, s.Type, s.Description)
	}
}
