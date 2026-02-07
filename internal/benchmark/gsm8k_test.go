package benchmark

import "testing"

func TestGSM8K_Evaluate(t *testing.T) {
	ds := &GSM8KDataset{}

	score, err := ds.Evaluate("The answer is 5.", "5")
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if score != 1 {
		t.Fatalf("score: got %v want %v", score, 1)
	}

	score, err = ds.Evaluate("6", "5")
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if score != 0 {
		t.Fatalf("score: got %v want %v", score, 0)
	}
}

func TestGSM8K_ExtractLastNumber(t *testing.T) {
	got, ok := extractLastNumber("Total: 1,234.")
	if !ok {
		t.Fatalf("extractLastNumber ok=false")
	}
	if got != "1234" {
		t.Fatalf("extractLastNumber: got %q want %q", got, "1234")
	}
}

