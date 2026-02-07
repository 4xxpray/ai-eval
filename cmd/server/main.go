package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/stellarlinkco/ai-eval/api"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

func main() {
	var addr string
	var configPath string
	flag.StringVar(&addr, "addr", ":8080", "listen address")
	flag.StringVar(&configPath, "config", config.DefaultPath, "path to config file")
	flag.Parse()

	cfg, err := config.Load(configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	st, err := store.Open(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer st.Close()

	lb, err := openLeaderboardStore(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer lb.Close()

	provider, err := llm.DefaultProviderFromConfig(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	srv, err := api.NewServer(cfg, st, provider, lb)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := srv.Run(addr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
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
		return leaderboard.NewStore(path)
	case "memory":
		return leaderboard.NewStore(":memory:")
	default:
		return nil, fmt.Errorf("leaderboard: unsupported type %q", storageType)
	}
}
