package store

import (
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/config"
)

const DefaultSQLitePath = "data/ai-eval.db"

func Open(cfg *config.Config) (Store, error) {
	if cfg == nil {
		return nil, fmt.Errorf("store: missing config")
	}

	storageType := strings.ToLower(strings.TrimSpace(cfg.Storage.Type))
	if storageType == "" {
		storageType = "sqlite"
	}

	switch storageType {
	case "sqlite":
		path := strings.TrimSpace(cfg.Storage.Path)
		if path == "" {
			path = DefaultSQLitePath
		}
		return NewSQLiteStore(path)
	case "memory":
		return NewSQLiteStore(":memory:")
	default:
		return nil, fmt.Errorf("store: unsupported type %q", storageType)
	}
}

