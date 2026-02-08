package store

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
)

func TestOpen_ErrorsAndTypes(t *testing.T) {
	if _, err := Open(nil); err == nil {
		t.Fatalf("Open(nil): expected error")
	}

	st, err := Open(&config.Config{Storage: config.StorageConfig{Type: "  ", Path: ":memory:"}})
	if err != nil {
		t.Fatalf("Open(default type sqlite): %v", err)
	}
	_ = st.Close()

	st, err = Open(&config.Config{Storage: config.StorageConfig{Type: "memory"}})
	if err != nil {
		t.Fatalf("Open(memory): %v", err)
	}
	_ = st.Close()

	st, err = Open(&config.Config{Storage: config.StorageConfig{Type: "sqlite", Path: ":memory:"}})
	if err != nil {
		t.Fatalf("Open(sqlite :memory:): %v", err)
	}
	_ = st.Close()

	if _, err := Open(&config.Config{Storage: config.StorageConfig{Type: "bad"}}); err == nil {
		t.Fatalf("Open(unsupported): expected error")
	}
}

func TestOpen_DefaultSQLitePath(t *testing.T) {
	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	tmp := t.TempDir()
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldWd)
	})

	st, err := Open(&config.Config{Storage: config.StorageConfig{Type: "sqlite", Path: "  "}})
	if err != nil {
		t.Fatalf("Open(default path): %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	if _, err := os.Stat(filepath.Join(tmp, DefaultSQLitePath)); err != nil {
		t.Fatalf("default db path: %v", err)
	}
}
