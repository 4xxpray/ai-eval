package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/stellarlinkco/ai-eval/api"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

var (
	osExit                 = os.Exit
	stderrWriter io.Writer = os.Stderr

	loadConfig                = config.Load
	openStore                 = store.Open
	defaultProviderFromConfig = llm.DefaultProviderFromConfig
	newServer                 = api.NewServer
	runServer                 = (*api.Server).Run

	leaderboardNewStore = leaderboard.NewStore
)

func main() {
	osExit(runMain(os.Args[1:]))
}

func runMain(args []string) int {
	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(stderrWriter)

	var addr string
	var configPath string
	fs.StringVar(&addr, "addr", ":8080", "listen address")
	fs.StringVar(&configPath, "config", config.DefaultPath, "path to config file")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	cfg, err := loadConfig(configPath)
	if err != nil {
		fmt.Fprintln(stderrWriter, err)
		return 1
	}

	st, err := openStore(cfg)
	if err != nil {
		fmt.Fprintln(stderrWriter, err)
		return 1
	}
	defer func() { _ = st.Close() }()

	lb, err := openLeaderboardStore(cfg)
	if err != nil {
		fmt.Fprintln(stderrWriter, err)
		return 1
	}
	defer func() { _ = lb.Close() }()

	provider, err := defaultProviderFromConfig(cfg)
	if err != nil {
		fmt.Fprintln(stderrWriter, err)
		return 1
	}

	srv, err := newServer(cfg, st, provider, lb)
	if err != nil {
		fmt.Fprintln(stderrWriter, err)
		return 1
	}
	if err := runServer(srv, addr); err != nil {
		fmt.Fprintln(stderrWriter, err)
		return 1
	}

	return 0
}

func openLeaderboardStore(cfg *config.Config) (*leaderboard.Store, error) {
	if cfg == nil {
		return nil, fmt.Errorf("leaderboard: missing config")
	}

	storageType := strings.ToLower(strings.TrimSpace(cfg.Storage.Type))
	if storageType == "" {
		storageType = "sqlite"
	}

	switch storageType {
	case "sqlite":
		path := strings.TrimSpace(cfg.Storage.Path)
		if path == "" {
			path = store.DefaultSQLitePath
		}
		return leaderboardNewStore(path)
	case "memory":
		return leaderboardNewStore(":memory:")
	default:
		return nil, fmt.Errorf("leaderboard: unsupported type %q", storageType)
	}
}
