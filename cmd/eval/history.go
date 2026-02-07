package main

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/spf13/cobra"
)

type historyOptions struct {
	promptName string
	limit      int
	since      string
}

func newHistoryCmd(st *cliState) *cobra.Command {
	var opts historyOptions

	cmd := &cobra.Command{
		Use:   "history",
		Short: "Show evaluation history",
		Args:  cobra.NoArgs,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(st.configPath)
			if err != nil {
				return err
			}
			st.cfg = cfg
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			return runHistoryList(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.promptName, "prompt", "", "prompt name to filter")
	cmd.Flags().IntVar(&opts.limit, "limit", 20, "max runs to list")
	cmd.Flags().StringVar(&opts.since, "since", "", "only runs since date (YYYY-MM-DD or RFC3339)")

	cmd.AddCommand(newHistoryShowCmd(st))
	return cmd
}

func newHistoryShowCmd(st *cliState) *cobra.Command {
	return &cobra.Command{
		Use:   "show <run-id>",
		Short: "Show details for a run",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runHistoryShow(cmd, st, args[0])
		},
	}
}

func runHistoryList(cmd *cobra.Command, st *cliState, opts *historyOptions) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("history: missing config (internal error)")
	}
	if opts == nil {
		return fmt.Errorf("history: nil options")
	}

	since, err := parseSince(opts.since)
	if err != nil {
		return err
	}

	stor, err := store.Open(st.cfg)
	if err != nil {
		return err
	}
	defer stor.Close()

	var reader store.RunReader = stor

	filter := store.RunFilter{
		PromptName: strings.TrimSpace(opts.promptName),
		Since:      since,
		Limit:      opts.limit,
	}
	runs, err := reader.ListRuns(cmd.Context(), filter)
	if err != nil {
		return err
	}

	out := cmd.OutOrStdout()
	if len(runs) == 0 {
		_, _ = fmt.Fprintln(out, "No runs found.")
		return nil
	}

	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "RUN_ID\tSTARTED\tFINISHED\tSUITES\tPASSED\tFAILED")
	for _, r := range runs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%d\t%d\n",
			r.ID,
			formatTime(r.StartedAt),
			formatTime(r.FinishedAt),
			r.TotalSuites,
			r.PassedSuites,
			r.FailedSuites,
		)
	}
	return tw.Flush()
}

func runHistoryShow(cmd *cobra.Command, st *cliState, runID string) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("history: missing config (internal error)")
	}

	runID = strings.TrimSpace(runID)
	if runID == "" {
		return fmt.Errorf("history: missing run id")
	}

	stor, err := store.Open(st.cfg)
	if err != nil {
		return err
	}
	defer stor.Close()

	var reader store.RunReader = stor

	run, err := reader.GetRun(cmd.Context(), runID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("history: run %q not found", runID)
		}
		return err
	}

	suites, err := reader.GetSuiteResults(cmd.Context(), runID)
	if err != nil {
		return err
	}

	out := cmd.OutOrStdout()
	_, _ = fmt.Fprintf(out, "Run: %s\n", run.ID)
	_, _ = fmt.Fprintf(out, "Started: %s\n", formatTime(run.StartedAt))
	_, _ = fmt.Fprintf(out, "Finished: %s\n", formatTime(run.FinishedAt))
	_, _ = fmt.Fprintf(out, "Suites: %d passed=%d failed=%d\n", run.TotalSuites, run.PassedSuites, run.FailedSuites)

	if len(suites) == 0 {
		return nil
	}

	for _, s := range suites {
		_, _ = fmt.Fprintf(out, "\nSuite: %s (prompt=%s version=%s)\n", s.SuiteName, s.PromptName, s.PromptVersion)
		_, _ = fmt.Fprintf(out, "Cases: %d passed=%d failed=%d pass_rate=%.2f avg_score=%.2f latency_ms=%d tokens=%d\n",
			s.TotalCases, s.PassedCases, s.FailedCases, s.PassRate, s.AvgScore, s.TotalLatency, s.TotalTokens)

		tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "CASE\tRESULT\tSCORE\tPASS@K\tLAT(ms)\tTOKENS\tERROR")
		for _, cr := range s.CaseResults {
			fmt.Fprintf(tw, "%s\t%s\t%.3f\t%.3f\t%d\t%d\t%s\n",
				cr.CaseID,
				statusLabel(cr.Passed),
				cr.Score,
				cr.PassAtK,
				cr.LatencyMs,
				cr.TokensUsed,
				cr.Error,
			)
		}
		_ = tw.Flush()
	}

	return nil
}

func parseSince(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, nil
	}
	layouts := []string{time.RFC3339, "2006-01-02"}
	for _, layout := range layouts {
		if ts, err := time.Parse(layout, s); err == nil {
			return ts, nil
		}
	}
	return time.Time{}, fmt.Errorf("history: invalid --since %q (expected YYYY-MM-DD or RFC3339)", s)
}

func formatTime(ts time.Time) string {
	if ts.IsZero() {
		return "-"
	}
	return ts.UTC().Format(time.RFC3339)
}

func statusLabel(passed bool) string {
	if passed {
		return "PASS"
	}
	return "FAIL"
}
