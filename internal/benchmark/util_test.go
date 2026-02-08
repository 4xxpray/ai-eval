package benchmark

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestReadJSONL_EmptyPath(t *testing.T) {
	_, err := readJSONL[struct{}](context.Background(), " \t\n ")
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "empty jsonl path") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestReadJSONL_File(t *testing.T) {
	type item struct {
		A int `json:"a"`
	}

	path := filepath.Join(t.TempDir(), "items.jsonl")
	if err := os.WriteFile(path, []byte("{\"a\":1}\n\n {\"a\":2}\n"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	got, err := readJSONL[item](context.Background(), path)
	if err != nil {
		t.Fatalf("readJSONL: %v", err)
	}
	if len(got) != 2 || got[0].A != 1 || got[1].A != 2 {
		t.Fatalf("got=%#v", got)
	}
}

func TestReadJSONL_Dir(t *testing.T) {
	type item struct {
		A int `json:"a"`
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "b.jsonl"), []byte("{\"a\":2}\n"), 0o644); err != nil {
		t.Fatalf("write b: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.jsonl"), []byte("{\"a\":1}\n"), 0o644); err != nil {
		t.Fatalf("write a: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "c.txt"), []byte("{\"a\":999}\n"), 0o644); err != nil {
		t.Fatalf("write txt: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "subdir"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	got, err := readJSONL[item](context.Background(), dir)
	if err != nil {
		t.Fatalf("readJSONL: %v", err)
	}
	if len(got) != 2 || got[0].A != 1 || got[1].A != 2 {
		t.Fatalf("got=%#v", got)
	}
}

func TestReadJSONL_StatError(t *testing.T) {
	_, err := readJSONL[struct{}](context.Background(), filepath.Join(t.TempDir(), "missing.jsonl"))
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestReadJSONL_OpenError(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod-based permission test is not reliable on Windows")
	}

	path := filepath.Join(t.TempDir(), "items.jsonl")
	if err := os.WriteFile(path, []byte("{\"a\":1}\n"), 0o000); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, err := readJSONL[struct{}](context.Background(), path)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestReadJSONLDir_ReadDirError(t *testing.T) {
	_, err := readJSONLDir[struct{}](context.Background(), filepath.Join(t.TempDir(), "missing"))
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestReadJSONLDir_FileError(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.jsonl"), []byte("{\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, err := readJSONLDir[struct{}](context.Background(), dir)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestDecodeJSONLStream_BadJSON(t *testing.T) {
	_, err := decodeJSONLStream[struct{}](context.Background(), strings.NewReader("{\n"))
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "parse jsonl") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestDecodeJSONLStream_ContextCanceledAfterFirstItem(t *testing.T) {
	type item struct {
		A int `json:"a"`
	}

	ctx := &errAfterNContext{
		Context: context.Background(),
		okCalls: 1,
		err:     context.Canceled,
	}

	got, err := decodeJSONLStream[item](ctx, strings.NewReader("{\"a\":1}\n{\"a\":2}\n"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
	if len(got) != 1 || got[0].A != 1 {
		t.Fatalf("got=%#v", got)
	}
}

func TestDecodeJSONLStream_ScannerErr(t *testing.T) {
	type item struct {
		A int `json:"a"`
	}

	wantErr := errors.New("boom")
	r := &errReader{
		data: []byte("{\"a\":1}\n"),
		err:  wantErr,
	}
	got, err := decodeJSONLStream[item](context.Background(), r)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v", err)
	}
	if len(got) != 1 || got[0].A != 1 {
		t.Fatalf("got=%#v", got)
	}
}

type errReader struct {
	data []byte
	err  error
	used bool
}

func (r *errReader) Read(p []byte) (int, error) {
	if !r.used {
		r.used = true
		n := copy(p, r.data)
		return n, nil
	}
	return 0, r.err
}

func TestTakeFirstN(t *testing.T) {
	in := []int{1, 2, 3}

	{
		out := takeFirstN(in, 0)
		if len(out) != 3 {
			t.Fatalf("len=%d", len(out))
		}
		if &out[0] != &in[0] {
			t.Fatalf("expected same slice when n<=0")
		}
	}

	{
		out := takeFirstN(in, len(in))
		if len(out) != 3 {
			t.Fatalf("len=%d", len(out))
		}
		if &out[0] != &in[0] {
			t.Fatalf("expected same slice when n>=len")
		}
	}

	{
		out := takeFirstN(in, 2)
		if len(out) != 2 || out[0] != 1 || out[1] != 2 {
			t.Fatalf("out=%v", out)
		}
		if &out[0] == &in[0] {
			t.Fatalf("expected copy when n<len")
		}
	}
}
