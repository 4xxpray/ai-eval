package benchmark

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGSM8KDataset_NameAndDescription(t *testing.T) {
	ds := &GSM8KDataset{}
	if ds.Name() != "gsm8k" {
		t.Fatalf("Name=%q", ds.Name())
	}
	if ds.Description() == "" {
		t.Fatalf("empty description")
	}
}

func TestGSM8KDataset_Load_NilContext(t *testing.T) {
	ds := &GSM8KDataset{}
	_, err := ds.Load(nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestGSM8KDataset_Load_MissingFile_DefaultSample(t *testing.T) {
	ds := &GSM8KDataset{SampleSize: 2}
	t.Setenv("AI_EVAL_GSM8K_PATH", filepath.Join(t.TempDir(), "missing.jsonl"))

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("len=%d", len(out))
	}
	if out[0].ID != "gsm8k-sample-1" {
		t.Fatalf("id=%q", out[0].ID)
	}
}

func TestGSM8KDataset_Load_FromFile_SkipsAndFallbacks(t *testing.T) {
	ds := &GSM8KDataset{}

	dir := t.TempDir()
	path := filepath.Join(dir, "gsm8k.jsonl")
	writeJSONLFile(t, path, []any{
		gsm8kRow{
			TaskID:   "t0",
			Question: " Q0 ",
			Answer:   "some #### 1,234",
		},
		gsm8kRow{
			Question: "  ",
			Answer:   "#### 999",
		},
		gsm8kRow{
			Question: "Q2",
			Answer:   "no marker 5",
		},
	})
	t.Setenv("AI_EVAL_GSM8K_PATH", path)

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("out=%#v", out)
	}
	if out[0].ID != "t0" || out[0].Answer != "1234" {
		t.Fatalf("row0=%#v", out[0])
	}
	if out[1].ID != "gsm8k-3" || out[1].Answer != "5" {
		t.Fatalf("row1=%#v", out[1])
	}
}

func TestGSM8KDataset_Load_AllSkipped_DefaultSample(t *testing.T) {
	ds := &GSM8KDataset{SampleSize: 1}
	dir := t.TempDir()
	path := filepath.Join(dir, "gsm8k.jsonl")
	writeJSONLFile(t, path, []any{
		gsm8kRow{Question: " ", Answer: "#### 1"},
	})
	t.Setenv("AI_EVAL_GSM8K_PATH", path)

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 1 || out[0].ID != "gsm8k-sample-1" {
		t.Fatalf("out=%#v", out)
	}
}

func TestGSM8KDataset_Load_ErrorWrap(t *testing.T) {
	ds := &GSM8KDataset{}
	dir := t.TempDir()
	path := filepath.Join(dir, "gsm8k.jsonl")
	if err := os.WriteFile(path, []byte("{\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv("AI_EVAL_GSM8K_PATH", path)

	_, err := ds.Load(context.Background())
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "gsm8k: load") || !strings.Contains(err.Error(), "parse jsonl") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestGSM8KDataset_Load_ContextErrorAfterScan(t *testing.T) {
	ds := &GSM8KDataset{}
	dir := t.TempDir()
	path := filepath.Join(dir, "gsm8k.jsonl")
	writeJSONLFile(t, path, []any{
		gsm8kRow{Question: "Q", Answer: "#### 1"},
	})
	t.Setenv("AI_EVAL_GSM8K_PATH", path)

	ctx := &errAfterNContext{
		Context: context.Background(),
		okCalls: 1,
		err:     context.Canceled,
	}
	_, err := ds.Load(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
}

func TestGSM8K_HelperFunctions(t *testing.T) {
	{
		_, ok := extractLastNumber("")
		if ok {
			t.Fatalf("expected ok=false for empty input")
		}
	}
	{
		_, ok := extractLastNumber(".")
		if ok {
			t.Fatalf("expected ok=false for dot input")
		}
	}
	{
		_, ok := extractLastNumber("-.")
		if ok {
			t.Fatalf("expected ok=false for dash-dot input")
		}
	}

	if got := extractExpectedNumber("x #### 7"); got != "7" {
		t.Fatalf("got=%q", got)
	}
	if got := extractExpectedNumber("no numbers here"); got != "no numbers here" {
		t.Fatalf("got=%q", got)
	}
	if _, ok := parseFloat(" "); ok {
		t.Fatalf("expected parseFloat to fail on blank")
	}
	if _, ok := parseFloat("x"); ok {
		t.Fatalf("expected parseFloat to fail on invalid")
	}
	if f, ok := parseFloat("1,234"); !ok || f != 1234 {
		t.Fatalf("parseFloat=%v ok=%v", f, ok)
	}
	if almostEqual(1, 2) {
		t.Fatalf("expected false")
	}
	if got := defaultGSM8KSample(); len(got) == 0 {
		t.Fatalf("default sample empty")
	}
}

func TestGSM8KDataset_Evaluate_ErrorPaths(t *testing.T) {
	ds := &GSM8KDataset{}

	{
		_, err := ds.Evaluate("5", "")
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := ds.Evaluate("5", "nope")
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := ds.Evaluate("no number", "5")
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := ds.Evaluate("1..2", "1")
		if err == nil {
			t.Fatalf("expected error")
		}
	}
}

func TestGSM8KDataset_Load_DefaultPathWhenEnvEmpty(t *testing.T) {
	ds := &GSM8KDataset{SampleSize: 1}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	tmpDir := t.TempDir()
	t.Cleanup(func() { _ = os.Chdir(cwd) })
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	if err := os.MkdirAll(filepath.Dir(defaultGSM8KPath), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	writeJSONLFile(t, defaultGSM8KPath, []any{
		gsm8kRow{Question: "Q", Answer: "#### 1"},
	})

	t.Setenv("AI_EVAL_GSM8K_PATH", " ")

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 1 || out[0].Answer != "1" {
		t.Fatalf("out=%#v", out)
	}
}
