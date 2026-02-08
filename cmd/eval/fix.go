package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"gopkg.in/yaml.v3"

	"github.com/spf13/cobra"
)

type fixOptions struct {
	promptPath string
	testsPath  string
	apply      bool
	dryRun     bool
}

func newFixCmd(st *cliState) *cobra.Command {
	var opts fixOptions

	cmd := &cobra.Command{
		Use:   "fix",
		Short: "Generate a fixed prompt based on diagnose suggestions",
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
			return runFix(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.promptPath, "prompt", "", "path to prompt file (.yaml/.yml or plain text); if omitted, read from stdin (cannot be used with --apply)")
	cmd.Flags().StringVar(&opts.testsPath, "tests", defaultTestsDir, "path to test suite file or directory")
	cmd.Flags().BoolVar(&opts.apply, "apply", false, "apply the fixed prompt back to --prompt file")
	cmd.Flags().BoolVar(&opts.dryRun, "dry-run", false, "print the fixed prompt but do not write any files")

	return cmd
}

func runFix(cmd *cobra.Command, st *cliState, opts *fixOptions) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("fix: missing config (internal error)")
	}
	if opts == nil {
		return fmt.Errorf("fix: nil options")
	}
	if opts.apply && strings.TrimSpace(opts.promptPath) == "" {
		return fmt.Errorf("fix: --apply requires --prompt (file path)")
	}

	provider, err := defaultProviderFromConfig(st.cfg)
	if err != nil {
		return fmt.Errorf("fix: %w", err)
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

	fixedPrompt := extractRewritePrompt(diag)
	if strings.TrimSpace(fixedPrompt) == "" {
		fixedPrompt, err = rewritePromptFallback(ctx, provider, pIn.PromptText, diag)
		if err != nil {
			return err
		}
	}

	if strings.TrimSpace(fixedPrompt) == "" {
		return errors.New("fix: empty fixed prompt (internal error)")
	}

	if !opts.apply || opts.dryRun {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), fixedPrompt)
	}

	if opts.apply && !opts.dryRun {
		if err := writeFixedPromptFunc(pIn, fixedPrompt); err != nil {
			return err
		}
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "\nApplied to: %s\n", strings.TrimSpace(pIn.Path))
	}

	return nil
}

func extractRewritePrompt(diag *optimizer.DiagnoseResult) string {
	if diag == nil {
		return ""
	}
	for _, s := range diag.Suggestions {
		if strings.EqualFold(strings.TrimSpace(s.Type), "rewrite_prompt") && strings.TrimSpace(s.After) != "" {
			return s.After
		}
	}
	// Allow slightly different naming.
	for _, s := range diag.Suggestions {
		if strings.Contains(strings.ToLower(strings.TrimSpace(s.Type)), "rewrite") && strings.TrimSpace(s.After) != "" {
			return s.After
		}
	}
	return ""
}

const rewritePromptFallbackTemplate = `You are a prompt engineer. Rewrite the prompt according to the diagnosis suggestions.

## Original Prompt
<prompt>
{{PROMPT}}
</prompt>

## Diagnosis (JSON)
{{DIAGNOSIS_JSON}}

## Output Format
Return ONLY valid JSON:
{"fixed_prompt": "the full fixed prompt"}`

func rewritePromptFallback(ctx context.Context, provider llm.Provider, promptText string, diag *optimizer.DiagnoseResult) (string, error) {
	if provider == nil {
		return "", errors.New("fix: nil llm provider")
	}
	promptText = strings.TrimSpace(promptText)
	if promptText == "" {
		return "", errors.New("fix: empty prompt")
	}
	if diag == nil {
		return "", errors.New("fix: nil diagnosis")
	}

	b, _ := json.Marshal(diag)

	p := strings.ReplaceAll(rewritePromptFallbackTemplate, "{{PROMPT}}", promptText)
	p = strings.ReplaceAll(p, "{{DIAGNOSIS_JSON}}", string(b))

	resp, err := provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: p}},
		MaxTokens: 8192,
	})
	if err != nil {
		return "", fmt.Errorf("fix: llm: %w", err)
	}
	if resp == nil {
		return "", errors.New("fix: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var parsed struct {
		FixedPrompt string `json:"fixed_prompt"`
	}
	if err := llm.ParseJSON(raw, &parsed); err != nil {
		return "", fmt.Errorf("fix: parse response: %w (response length: %d)", err, len(raw))
	}
	return strings.TrimSpace(parsed.FixedPrompt), nil
}

func writeFixedPrompt(pIn *promptInput, fixedPrompt string) error {
	if pIn == nil {
		return errors.New("fix: nil prompt input")
	}
	path := strings.TrimSpace(pIn.Path)
	if path == "" {
		return errors.New("fix: missing prompt path")
	}
	if strings.TrimSpace(fixedPrompt) == "" {
		return errors.New("fix: empty fixed prompt")
	}

	if pIn.IsYAML {
		if pIn.Prompt == nil {
			return errors.New("fix: nil yaml prompt")
		}
		p := *pIn.Prompt
		p.Template = fixedPrompt

		b, err := yaml.Marshal(&p)
		if err != nil {
			return fmt.Errorf("fix: marshal yaml: %w", err)
		}
		if err := os.WriteFile(path, b, 0o644); err != nil {
			return fmt.Errorf("fix: write %q: %w", path, err)
		}
		return nil
	}

	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".yaml" || ext == ".yml" {
		return fmt.Errorf("fix: refusing to overwrite %q as plain text (looks like yaml)", path)
	}

	if err := os.WriteFile(path, []byte(fixedPrompt), 0o644); err != nil {
		return fmt.Errorf("fix: write %q: %w", path, err)
	}
	return nil
}

var writeFixedPromptFunc = writeFixedPrompt
