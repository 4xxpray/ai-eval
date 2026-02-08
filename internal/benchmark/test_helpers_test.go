package benchmark

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeJSONLFile(t *testing.T, path string, lines []any) {
	t.Helper()

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, line := range lines {
		if err := enc.Encode(line); err != nil {
			t.Fatalf("encode line %d: %v", i, err)
		}
	}
}

func writeExecutable(t *testing.T, dir string, name string, content string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

type errAfterNContext struct {
	context.Context
	okCalls int
	err     error
	calls   int
}

func (c *errAfterNContext) Err() error {
	c.calls++
	if c.calls <= c.okCalls {
		return nil
	}
	return c.err
}
