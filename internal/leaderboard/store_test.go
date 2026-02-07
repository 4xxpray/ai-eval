package leaderboard

import (
	"context"
	"testing"
	"time"
)

func TestStore_SaveAndGetLeaderboard(t *testing.T) {
	st, err := NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer st.Close()

	ctx := context.Background()
	e1 := &Entry{
		Model:    "m1",
		Provider: "openai",
		Dataset:  "mmlu",
		Score:    0.80,
		Accuracy: 0.80,
		Latency:  120,
		Cost:     0,
		EvalDate: time.UnixMilli(1000).UTC(),
	}
	e2 := &Entry{
		Model:    "m2",
		Provider: "openai",
		Dataset:  "mmlu",
		Score:    0.90,
		Accuracy: 0.90,
		Latency:  200,
		Cost:     0,
		EvalDate: time.UnixMilli(2000).UTC(),
	}

	if err := st.Save(ctx, e1); err != nil {
		t.Fatalf("Save e1: %v", err)
	}
	if err := st.Save(ctx, e2); err != nil {
		t.Fatalf("Save e2: %v", err)
	}
	if e1.ID == 0 || e2.ID == 0 {
		t.Fatalf("expected IDs to be set (got e1=%d e2=%d)", e1.ID, e2.ID)
	}

	got, err := st.GetLeaderboard(ctx, "mmlu", 10)
	if err != nil {
		t.Fatalf("GetLeaderboard: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(entries): got %d want %d", len(got), 2)
	}
	if got[0].Model != "m2" {
		t.Fatalf("rank1 model: got %q want %q", got[0].Model, "m2")
	}
	if got[1].Model != "m1" {
		t.Fatalf("rank2 model: got %q want %q", got[1].Model, "m1")
	}
}

func TestStore_GetModelHistory_Order(t *testing.T) {
	st, err := NewStore(":memory:")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer st.Close()

	ctx := context.Background()
	if err := st.Save(ctx, &Entry{
		Model:    "m1",
		Provider: "claude",
		Dataset:  "gsm8k",
		Score:    0.20,
		Accuracy: 0.20,
		Latency:  10,
		Cost:     0,
		EvalDate: time.UnixMilli(1000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := st.Save(ctx, &Entry{
		Model:    "m1",
		Provider: "claude",
		Dataset:  "gsm8k",
		Score:    0.90,
		Accuracy: 0.90,
		Latency:  20,
		Cost:     0,
		EvalDate: time.UnixMilli(2000).UTC(),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := st.GetModelHistory(ctx, "m1", "gsm8k")
	if err != nil {
		t.Fatalf("GetModelHistory: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(history): got %d want %d", len(got), 2)
	}
	if got[0].Score != 0.90 {
		t.Fatalf("history[0].Score: got %.2f want %.2f", got[0].Score, 0.90)
	}
	if got[1].Score != 0.20 {
		t.Fatalf("history[1].Score: got %.2f want %.2f", got[1].Score, 0.20)
	}
}

