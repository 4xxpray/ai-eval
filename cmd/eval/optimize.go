package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/generator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/spf13/cobra"
)

func newOptimizeCmd(st *cliState) *cobra.Command {
	var (
		promptFile   string
		outputFile   string
		numCases     int
		maxIter      int
		showProgress bool
		varFlags     []string
	)

	cmd := &cobra.Command{
		Use:   "optimize",
		Short: "Automatically evaluate and optimize a prompt",
		Long: `Analyze a prompt, generate test cases, run evaluation, and output an optimized version.

Examples:
  ai-eval optimize --prompt prompt.txt
  ai-eval optimize --prompt prompt.txt --output optimized.txt
  cat prompt.txt | ai-eval optimize
`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(st.configPath)
			if err != nil {
				return fmt.Errorf("failed to load config: %w", err)
			}

			var promptContent string

			if promptFile != "" {
				data, err := os.ReadFile(promptFile)
				if err != nil {
					return fmt.Errorf("failed to read prompt file: %w", err)
				}
				promptContent = string(data)
			} else {
				stat, _ := os.Stdin.Stat()
				if (stat.Mode() & os.ModeCharDevice) == 0 {
					data, err := io.ReadAll(os.Stdin)
					if err != nil {
						return fmt.Errorf("failed to read from stdin: %w", err)
					}
					promptContent = string(data)
				} else {
					return errors.New("no prompt provided: use --prompt or pipe content to stdin")
				}
			}

			promptContent = strings.TrimSpace(promptContent)
			if promptContent == "" {
				return errors.New("prompt content is empty")
			}

			provider, err := llm.DefaultProviderFromConfig(cfg)
			if err != nil {
				return err
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			promptName := "prompt"
			if promptFile != "" {
				promptName = strings.TrimSuffix(filepath.Base(promptFile), filepath.Ext(promptFile))
			}

			// Parse template variables from --var flags
			variables := make(map[string]string)
			for _, v := range varFlags {
				parts := strings.SplitN(v, "=", 2)
				if len(parts) == 2 {
					variables[parts[0]] = parts[1]
				}
			}

			if showProgress {
				fmt.Println("🔍 Analyzing prompt and generating test cases...")
			}

			gen := &generator.Generator{Provider: provider}
			genResult, err := gen.Generate(ctx, &generator.GenerateRequest{
				PromptContent: promptContent,
				PromptName:    promptName,
				NumCases:      numCases,
				Variables:     variables,
			})
			if err != nil {
				return fmt.Errorf("failed to generate test cases: %w", err)
			}

			if showProgress {
				fmt.Printf("📋 Generated %d test cases\n", len(genResult.Suite.Cases))
				if genResult.IsSystemPrompt {
					fmt.Println("📌 Detected as SYSTEM PROMPT (will use as system message)")
				}
				fmt.Println("\n📊 Analysis:")
				fmt.Println(genResult.Analysis)
				fmt.Println("\n🚀 Running evaluation...")
			}

			p := &prompt.Prompt{
				Name:           promptName,
				Template:       promptContent,
				IsSystemPrompt: genResult.IsSystemPrompt,
			}

			registry := evaluator.NewRegistry()
			r := runner.NewRunner(provider, registry, runner.Config{
				Trials:        1,
				Concurrency:   1,
				PassThreshold: 0.6,
				Timeout:       2 * time.Minute,
			})

			suiteResult, err := r.RunSuite(ctx, p, genResult.Suite)
			if err != nil {
				return fmt.Errorf("failed to run evaluation: %w", err)
			}

			if showProgress {
				fmt.Printf("\n📈 Evaluation Results: %.1f%% pass rate (%.2f avg score)\n",
					suiteResult.PassRate*100, suiteResult.AvgScore)
			}

			if suiteResult.PassRate >= 0.9 && suiteResult.AvgScore >= 0.9 {
				if showProgress {
					fmt.Println("\n✅ Prompt is already performing well! No optimization needed.")
				}
				fmt.Println("\n--- Original Prompt (No Changes) ---")
				fmt.Println(promptContent)
				return nil
			}

			if showProgress {
				fmt.Println("\n🔧 Optimizing prompt based on evaluation results...")
			}

			opt := &optimizer.Optimizer{Provider: provider}
			optResult, err := opt.Optimize(ctx, &optimizer.OptimizeRequest{
				OriginalPrompt: promptContent,
				EvalResults:    suiteResult,
				MaxIterations:  maxIter,
			})
			if err != nil {
				return fmt.Errorf("failed to optimize prompt: %w", err)
			}

			if showProgress {
				fmt.Println("\n📝 Optimization Summary:")
				fmt.Println(optResult.Summary)
				fmt.Println("\n🔄 Changes Made:")
				for i, c := range optResult.Changes {
					fmt.Printf("  %d. [%s] %s\n", i+1, c.Type, c.Description)
				}
			}

			if outputFile != "" {
				if err := os.WriteFile(outputFile, []byte(optResult.OptimizedPrompt), 0644); err != nil {
					return fmt.Errorf("failed to write output file: %w", err)
				}
				if showProgress {
					fmt.Printf("\n✅ Optimized prompt saved to: %s\n", outputFile)
				}
			} else {
				fmt.Println("\n--- Optimized Prompt ---")
				fmt.Println(optResult.OptimizedPrompt)
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&promptFile, "prompt", "p", "", "path to prompt file")
	cmd.Flags().StringVarP(&outputFile, "output", "o", "", "output file for optimized prompt")
	cmd.Flags().IntVarP(&numCases, "cases", "c", 5, "number of test cases to generate")
	cmd.Flags().IntVar(&maxIter, "iterations", 1, "max optimization iterations")
	cmd.Flags().BoolVar(&showProgress, "progress", true, "show progress messages")
	cmd.Flags().StringArrayVar(&varFlags, "var", nil, "template variable in KEY=VALUE format (can be repeated)")

	return cmd
}
