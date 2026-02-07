package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"text/tabwriter"

	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/spf13/cobra"
)

type leaderboardOptions struct {
	dataset string
	top     int
	format  string
}

func newLeaderboardCmd(st *cliState) *cobra.Command {
	var opts leaderboardOptions

	cmd := &cobra.Command{
		Use:   "leaderboard",
		Short: "Show benchmark leaderboard",
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
			return runLeaderboard(cmd, st, &opts)
		},
	}

	cmd.Flags().StringVar(&opts.dataset, "dataset", "", "dataset name")
	cmd.Flags().IntVar(&opts.top, "top", 20, "top N entries")
	cmd.Flags().StringVar(&opts.format, "format", "table", "output format: table|json")

	return cmd
}

func runLeaderboard(cmd *cobra.Command, st *cliState, opts *leaderboardOptions) error {
	if st == nil || st.cfg == nil {
		return fmt.Errorf("leaderboard: missing config (internal error)")
	}
	if opts == nil {
		return fmt.Errorf("leaderboard: nil options")
	}

	ds := strings.TrimSpace(opts.dataset)
	if ds == "" {
		return fmt.Errorf("leaderboard: missing --dataset")
	}

	lb, err := openLeaderboardStore(st.cfg)
	if err != nil {
		return err
	}
	defer lb.Close()

	entries, err := lb.GetLeaderboard(cmd.Context(), ds, opts.top)
	if err != nil {
		return err
	}

	switch strings.ToLower(strings.TrimSpace(opts.format)) {
	case "", "table":
		tw := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "RANK\tMODEL\tPROVIDER\tSCORE\tACCURACY\tLAT(ms)\tCOST\tDATE")
		for i, e := range entries {
			fmt.Fprintf(tw, "%d\t%s\t%s\t%.4f\t%.4f\t%d\t%.4f\t%s\n",
				i+1,
				e.Model,
				e.Provider,
				e.Score,
				e.Accuracy,
				e.Latency,
				e.Cost,
				e.EvalDate.UTC().Format("2006-01-02 15:04:05Z"),
			)
		}
		return tw.Flush()
	case "json":
		enc := json.NewEncoder(cmd.OutOrStdout())
		enc.SetIndent("", "  ")
		return enc.Encode(entries)
	default:
		return fmt.Errorf("leaderboard: invalid --format %q (expected table|json)", opts.format)
	}
}
