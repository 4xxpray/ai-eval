package main

import (
	"fmt"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/app"
)

func newListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List prompts or test suites",
		Args:  cobra.NoArgs,
	}

	cmd.AddCommand(newListPromptsCmd())
	cmd.AddCommand(newListTestsCmd())
	return cmd
}

func newListPromptsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "prompts",
		Short: "List available prompts",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			prompts, err := app.LoadPrompts(defaultPromptsDir)
			if err != nil {
				return err
			}
			sort.Slice(prompts, func(i, j int) bool {
				return strings.ToLower(prompts[i].Name) < strings.ToLower(prompts[j].Name)
			})

			tw := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(tw, "NAME\tVERSION\tDESCRIPTION")
			for _, p := range prompts {
				fmt.Fprintf(tw, "%s\t%s\t%s\n", p.Name, p.Version, p.Description)
			}
			return tw.Flush()
		},
	}
}

func newListTestsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "tests",
		Short: "List available test suites",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			suites, err := app.LoadTestSuites(defaultTestsDir)
			if err != nil {
				return err
			}
			sort.Slice(suites, func(i, j int) bool {
				return strings.ToLower(suites[i].Suite) < strings.ToLower(suites[j].Suite)
			})

			tw := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(tw, "SUITE\tPROMPT\tCASES\tDESCRIPTION")
			for _, s := range suites {
				fmt.Fprintf(tw, "%s\t%s\t%d\t%s\n", s.Suite, s.Prompt, len(s.Cases), s.Description)
			}
			return tw.Flush()
		},
	}
}
