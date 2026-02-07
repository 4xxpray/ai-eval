package benchmark

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func readJSONL[T any](ctx context.Context, path string) ([]T, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, errors.New("benchmark: empty jsonl path")
	}

	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return readJSONLDir[T](ctx, path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return decodeJSONLStream[T](ctx, f)
}

func readJSONLDir[T any](ctx context.Context, dir string) ([]T, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var paths []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := strings.ToLower(strings.TrimSpace(e.Name()))
		if !strings.HasSuffix(name, ".jsonl") {
			continue
		}
		paths = append(paths, filepath.Join(dir, e.Name()))
	}
	sort.Strings(paths)

	var out []T
	for _, p := range paths {
		items, err := readJSONL[T](ctx, p)
		if err != nil {
			return nil, err
		}
		out = append(out, items...)
	}
	return out, nil
}

func decodeJSONLStream[T any](ctx context.Context, r io.Reader) ([]T, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var out []T
	for sc.Scan() {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}

		var item T
		if err := json.Unmarshal(line, &item); err != nil {
			return out, fmt.Errorf("benchmark: parse jsonl: %w", err)
		}
		out = append(out, item)
	}
	if err := sc.Err(); err != nil {
		return out, err
	}
	return out, nil
}

func takeFirstN[T any](in []T, n int) []T {
	if n <= 0 || n >= len(in) {
		return in
	}
	out := make([]T, 0, n)
	return append(out, in[:n]...)
}

