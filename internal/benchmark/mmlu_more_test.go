package benchmark

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMMLUDataset_NameAndDescription(t *testing.T) {
	ds := &MMLUDataset{}
	if ds.Name() != "mmlu" {
		t.Fatalf("Name=%q", ds.Name())
	}
	if ds.Description() == "" {
		t.Fatalf("empty description")
	}
}

func TestMMLUDataset_Load_NilContext(t *testing.T) {
	ds := &MMLUDataset{}
	_, err := ds.Load(nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestMMLUDataset_Load_MissingFile_DefaultSample(t *testing.T) {
	ds := &MMLUDataset{SampleSize: 1}
	t.Setenv("AI_EVAL_MMLU_PATH", filepath.Join(t.TempDir(), "missing.jsonl"))

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len=%d", len(out))
	}
	if out[0].ID != "mmlu-sample-1" {
		t.Fatalf("id=%q", out[0].ID)
	}
}

func TestMMLUDataset_Load_FromFile_FilterAndDefaults(t *testing.T) {
	ds := &MMLUDataset{Subjects: []string{"math"}}

	dir := t.TempDir()
	path := filepath.Join(dir, "mmlu.jsonl")
	writeJSONLFile(t, path, []any{
		mmluRow{
			TaskID:   "task-0",
			Question: " Q0 ",
			Choices:  []string{"a", " ", "b"},
			Answer:   "B",
			Subject:  "Math",
		},
		mmluRow{
			Question: "Q1",
			Answer:   2,
			Subject:  "math",
			Category: "cat",
		},
		mmluRow{
			Question: "ignore",
			Answer:   "A",
			Subject:  "history",
		},
	})
	t.Setenv("AI_EVAL_MMLU_PATH", path)

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("out=%#v", out)
	}
	if out[0].ID != "task-0" || out[0].Category != "Math" {
		t.Fatalf("row0=%#v", out[0])
	}
	if out[1].ID != "mmlu-2" || out[1].Category != "cat" {
		t.Fatalf("row1=%#v", out[1])
	}
	if len(out[0].Choices) != 2 || out[0].Choices[0] != "a" || out[0].Choices[1] != "b" {
		t.Fatalf("choices=%#v", out[0].Choices)
	}

	exp, ok := out[0].Answer.(mcqExpected)
	if !ok {
		t.Fatalf("answer type=%T", out[0].Answer)
	}
	if exp.Answer != "B" || len(exp.Choices) != 2 {
		t.Fatalf("expected=%#v", exp)
	}
}

func TestMMLUDataset_Load_AllSkipped_DefaultSample(t *testing.T) {
	ds := &MMLUDataset{Subjects: []string{"math"}, SampleSize: 2}

	dir := t.TempDir()
	path := filepath.Join(dir, "mmlu.jsonl")
	writeJSONLFile(t, path, []any{
		mmluRow{Question: "Q", Answer: "A", Subject: "history"},
		mmluRow{Question: " ", Answer: "B", Subject: "math"},
	})
	t.Setenv("AI_EVAL_MMLU_PATH", path)

	out, err := ds.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("len=%d", len(out))
	}
	if out[0].ID != "mmlu-sample-1" {
		t.Fatalf("id=%q", out[0].ID)
	}
}

func TestMMLU_HelperFunctions(t *testing.T) {
	choices := []string{"Earth", "Mars", "Jupiter", "Venus"}

	if _, err := normalizeIndex(0, 4); err != nil {
		t.Fatalf("normalizeIndex 0-based: %v", err)
	}
	if idx, err := normalizeIndex(1, 4); err != nil || idx != 1 {
		t.Fatalf("normalizeIndex idx=1: idx=%d err=%v", idx, err)
	}
	if idx, err := normalizeIndex(4, 4); err != nil || idx != 3 {
		t.Fatalf("normalizeIndex 1-based max: idx=%d err=%v", idx, err)
	}
	if _, err := normalizeIndex(5, 4); err == nil {
		t.Fatalf("normalizeIndex out of range: expected error")
	}

	if idx, err := expectedChoiceIndex(int64(1), choices); err != nil || idx != 1 {
		t.Fatalf("expectedChoiceIndex(int64): idx=%d err=%v", idx, err)
	}
	if idx, err := expectedChoiceIndex(float64(2), choices); err != nil || idx != 2 {
		t.Fatalf("expectedChoiceIndex(float64): idx=%d err=%v", idx, err)
	}
	if idx, err := expectedChoiceIndex("b", choices); err != nil || idx != 1 {
		t.Fatalf("expectedChoiceIndex(letter): idx=%d err=%v", idx, err)
	}
	if idx, err := expectedChoiceIndex("2", choices); err != nil || idx != 2 {
		t.Fatalf("expectedChoiceIndex(number): idx=%d err=%v", idx, err)
	}
	if idx, err := expectedChoiceIndex("Mars", choices); err != nil || idx != 1 {
		t.Fatalf("expectedChoiceIndex(choice): idx=%d err=%v", idx, err)
	}
	if _, err := expectedChoiceIndex(true, choices); err == nil {
		t.Fatalf("expectedChoiceIndex(unsupported): expected error")
	}

	if _, err := parseExpectedString("", choices, 4); err == nil {
		t.Fatalf("parseExpectedString(empty): expected error")
	}
	if idx, err := parseExpectedString("Mars", choices, 4); err != nil || idx != 1 {
		t.Fatalf("parseExpectedString(text): idx=%d err=%v", idx, err)
	}
	if _, err := parseExpectedString("nope", choices, 4); err == nil {
		t.Fatalf("parseExpectedString(no match): expected error")
	}

	{
		a, c := unwrapMCQExpected(mcqExpected{Answer: "B", Choices: choices})
		if a != "B" || len(c) != 4 {
			t.Fatalf("unwrap=%v/%v", a, c)
		}
	}
	{
		a, c := unwrapMCQExpected(&mcqExpected{Answer: "A", Choices: choices})
		if a != "A" || len(c) != 4 {
			t.Fatalf("unwrap ptr=%v/%v", a, c)
		}
	}
	{
		a, c := unwrapMCQExpected((*mcqExpected)(nil))
		if a != nil || c != nil {
			t.Fatalf("unwrap nil ptr=%v/%v", a, c)
		}
	}
	{
		a, c := unwrapMCQExpected("raw")
		if a != "raw" || c != nil {
			t.Fatalf("unwrap default=%v/%v", a, c)
		}
	}

	if got := normalizeStringSet([]string{" ", "A", "a"}); len(got) != 1 || !got["a"] {
		t.Fatalf("normalizeStringSet=%#v", got)
	}
	if got := normalizeStringSet([]string{" ", "\t"}); got != nil {
		t.Fatalf("normalizeStringSet blank=%#v", got)
	}

	if got := compactStrings([]string{"", " a ", " "}); len(got) != 1 || got[0] != "a" {
		t.Fatalf("compactStrings=%#v", got)
	}
	if got := compactStrings(nil); got != nil {
		t.Fatalf("compactStrings(nil)=%#v", got)
	}
	if got := defaultMMLUSample(); len(got) == 0 {
		t.Fatalf("defaultMMLUSample empty")
	}

	// Exercise json.Number path via expectedChoiceIndex on decode output.
	var v any
	if err := json.Unmarshal([]byte(`1`), &v); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if _, err := expectedChoiceIndex(v, nil); err != nil {
		t.Fatalf("expectedChoiceIndex(default choices): %v", err)
	}
}

func TestMMLU_ParseMCQResponse_EmptyChoicesAndTextMatch(t *testing.T) {
	{
		_, ok := parseMCQResponse("A", nil)
		if !ok {
			t.Fatalf("expected ok for letter token")
		}
	}
	{
		idx, ok := parseMCQResponse("I think it's Mars.", []string{"Earth", "Mars"})
		if !ok || idx != 1 {
			t.Fatalf("idx=%d ok=%v", idx, ok)
		}
	}
	{
		_, ok := parseMCQResponse("Z", []string{"A"})
		if ok {
			t.Fatalf("expected false")
		}
	}
	{
		choices := make([]string, 30)
		for i := range choices {
			choices[i] = "x"
		}
		idx, ok := parseMCQResponse("A", choices)
		if !ok || idx != 0 {
			t.Fatalf("idx=%d ok=%v", idx, ok)
		}
	}
}

func TestMMLUDataset_Load_ErrorWrap(t *testing.T) {
	ds := &MMLUDataset{}
	dir := t.TempDir()
	path := filepath.Join(dir, "mmlu.jsonl")
	if err := os.WriteFile(path, []byte("{\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv("AI_EVAL_MMLU_PATH", path)
	_, err := ds.Load(context.Background())
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "mmlu: load") {
		t.Fatalf("err=%q", err.Error())
	}
	if !strings.Contains(err.Error(), "parse jsonl") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestMMLUDataset_Evaluate_ErrorPaths(t *testing.T) {
	ds := &MMLUDataset{}

	{
		_, err := ds.Evaluate("A", mcqExpected{Answer: true})
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := ds.Evaluate("", mcqExpected{Answer: "A"})
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := ds.Evaluate("A", mcqExpected{Answer: ""})
		if err == nil {
			t.Fatalf("expected error")
		}
	}
}

func TestMMLU_IsAlphaNum(t *testing.T) {
	if !isAlphaNum('a') || !isAlphaNum('Z') || !isAlphaNum('0') {
		t.Fatalf("expected alnum")
	}
	if isAlphaNum(' ') {
		t.Fatalf("expected non-alnum")
	}
}

func TestMMLU_ParseExpectedString_IntAndLetterOutOfRange(t *testing.T) {
	choices := []string{"A"}
	if _, err := parseExpectedString("B", choices, 1); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := parseExpectedString("2", choices, 1); err == nil {
		t.Fatalf("expected error")
	}
}

func TestMMLU_expectedChoiceIndex_DefaultChoicesMax26(t *testing.T) {
	choices := make([]string, 30)
	for i := range choices {
		choices[i] = "x"
	}
	if idx, err := expectedChoiceIndex(25, choices); err != nil || idx != 25 {
		t.Fatalf("idx=%d err=%v", idx, err)
	}
	if idx, err := expectedChoiceIndex(26, choices); err != nil || idx != 25 {
		t.Fatalf("idx=%d err=%v", idx, err)
	}
	if _, err := expectedChoiceIndex(27, choices); err == nil {
		t.Fatalf("expected error")
	}
}

func TestMMLU_matchChoiceText_MaxBoundary(t *testing.T) {
	if idx, ok := matchChoiceText("venus", []string{"Earth", "Mars", "Venus"}, 2); ok || idx != -1 {
		t.Fatalf("idx=%d ok=%v", idx, ok)
	}
	if _, ok := matchChoiceText("x", nil, 4); ok {
		t.Fatalf("expected false")
	}
	if idx, ok := matchChoiceText("mars", []string{"", "Mars"}, 4); !ok || idx != 1 {
		t.Fatalf("idx=%d ok=%v", idx, ok)
	}
}

func TestMMLUDataset_Load_ContextErrorAfterScan(t *testing.T) {
	ds := &MMLUDataset{}
	dir := t.TempDir()
	path := filepath.Join(dir, "mmlu.jsonl")
	writeJSONLFile(t, path, []any{
		mmluRow{Question: "Q", Choices: []string{"A"}, Answer: "A", Subject: "math"},
	})
	t.Setenv("AI_EVAL_MMLU_PATH", path)

	ctx := &errAfterNContext{
		Context: context.Background(),
		okCalls: 1, // 1 line scanned
		err:     context.Canceled,
	}
	_, err := ds.Load(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
}

func TestMMLU_ExtractNumberToken_More(t *testing.T) {
	{
		idx, ok := extractNumberToken("0", 4)
		if !ok || idx != 0 {
			t.Fatalf("idx=%d ok=%v", idx, ok)
		}
	}
	{
		idx, ok := extractNumberToken("9 2", 4)
		if !ok || idx != 1 {
			t.Fatalf("idx=%d ok=%v", idx, ok)
		}
	}
	{
		idx, ok := extractNumberToken("999999999999999999999 2", 4)
		if !ok || idx != 1 {
			t.Fatalf("idx=%d ok=%v", idx, ok)
		}
	}
}
