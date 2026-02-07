package benchmark

import "testing"

func TestMMLU_ParseMCQResponse(t *testing.T) {
	choices := []string{"Earth", "Mars", "Jupiter", "Venus"}
	tests := []struct {
		in   string
		want int
		ok   bool
	}{
		{in: "B", want: 1, ok: true},
		{in: "Answer: (C)", want: 2, ok: true},
		{in: "2", want: 1, ok: true},
		{in: "Mars", want: 1, ok: true},
		{in: "", ok: false},
	}

	for _, tc := range tests {
		got, ok := parseMCQResponse(tc.in, choices)
		if ok != tc.ok {
			t.Fatalf("parseMCQResponse(%q): ok=%v want %v", tc.in, ok, tc.ok)
		}
		if tc.ok && got != tc.want {
			t.Fatalf("parseMCQResponse(%q): got %d want %d", tc.in, got, tc.want)
		}
	}
}

func TestMMLU_Evaluate(t *testing.T) {
	ds := &MMLUDataset{}
	expected := mcqExpected{
		Answer:  "B",
		Choices: []string{"Earth", "Mars", "Jupiter", "Venus"},
	}

	score, err := ds.Evaluate("B", expected)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if score != 1 {
		t.Fatalf("score: got %v want %v", score, 1)
	}

	score, err = ds.Evaluate("A", expected)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if score != 0 {
		t.Fatalf("score: got %v want %v", score, 0)
	}
}

