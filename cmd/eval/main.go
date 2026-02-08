package main

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
	"github.com/stellarlinkco/ai-eval/internal/config"
)

const (
	defaultPromptsDir = "prompts"
	defaultTestsDir   = "tests"
)

type cliState struct {
	configPath string
	cfg        *config.Config
}

var (
	osExit                 = os.Exit
	stderrWriter io.Writer = os.Stderr
)

func main() {
	cmd := newRootCmd()
	if err := cmd.Execute(); err != nil {
		if errors.Is(err, errTestsFailed) || errors.Is(err, errRegression) {
			osExit(1)
			return
		}
		fmt.Fprintln(stderrWriter, err)
		osExit(1)
	}
}

func newRootCmd() *cobra.Command {
	st := &cliState{configPath: config.DefaultPath}

	root := &cobra.Command{
		Use:           "ai-eval",
		Short:         "Run prompt evaluation suites",
		SilenceErrors: true,
		SilenceUsage:  true,
	}
	root.PersistentFlags().StringVar(&st.configPath, "config", st.configPath, "path to config file")

	root.AddCommand(newRunCmd(st))
	root.AddCommand(newCompareCmd(st))
	root.AddCommand(newRedteamCmd(st))
	root.AddCommand(newListCmd())
	root.AddCommand(newHistoryCmd(st))
	root.AddCommand(newOptimizeCmd(st))
	root.AddCommand(newDiagnoseCmd(st))
	root.AddCommand(newFixCmd(st))
	root.AddCommand(newBenchmarkCmd(st))
	root.AddCommand(newLeaderboardCmd(st))
	return root
}
