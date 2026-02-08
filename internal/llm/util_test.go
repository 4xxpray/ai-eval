package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestText(t *testing.T) {
	t.Parallel()

	if got := Text(nil); got != "" {
		t.Fatalf("Text(nil): got %q want %q", got, "")
	}

	resp := &Response{
		Content: []ContentBlock{
			{Type: "tool_use", Text: "ignored"},
			{Type: "text", Text: "a"},
			{Type: "text", Text: "b"},
			{Type: "tool_use", Text: "ignored2"},
		},
	}
	if got := Text(resp); got != "ab" {
		t.Fatalf("Text(resp): got %q want %q", got, "ab")
	}
}

func TestParseJSON(t *testing.T) {
	t.Parallel()

	type outT struct {
		A int `json:"a"`
	}

	tests := []struct {
		name    string
		raw     string
		wantA   int
		wantErr string
	}{
		{name: "Empty", raw: " \n\t ", wantErr: "empty output"},
		{name: "MissingObject", raw: "nope", wantErr: "missing JSON object"},
		{name: "InvalidJSON", raw: `{"a":}`, wantErr: "invalid character"},
		{name: "PlainJSON", raw: `{"a":1}`, wantA: 1},
		{name: "WrappedInText", raw: `prefix {"a":2} suffix`, wantA: 2},
		{name: "FencedJSON", raw: "```json\n{\"a\":3}\n```", wantA: 3},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var out outT
			err := ParseJSON(tt.raw, &out)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("ParseJSON: expected error")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error: got %q want contains %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseJSON: %v", err)
			}
			if out.A != tt.wantA {
				t.Fatalf("out.A: got %d want %d", out.A, tt.wantA)
			}
		})
	}

	t.Run("JSONNumber", func(t *testing.T) {
		t.Parallel()

		var out struct {
			A json.Number `json:"a"`
		}
		if err := ParseJSON(`{"a": 4}`, &out); err != nil {
			t.Fatalf("ParseJSON: %v", err)
		}
		if out.A.String() != "4" {
			t.Fatalf("out.A: got %q want %q", out.A.String(), "4")
		}
	})
}
