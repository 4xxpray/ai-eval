package claude

import "testing"

func TestClaudeText(t *testing.T) {
	t.Parallel()

	if got := ClaudeText(nil); got != "" {
		t.Fatalf("ClaudeText(nil): got %q want empty", got)
	}

	got := ClaudeText(&Response{
		Content: []ContentBlock{
			{Type: "tool_use", Name: "x"},
			{Type: "text", Text: "a"},
			{Type: "text", Text: "b"},
		},
	})
	if got != "ab" {
		t.Fatalf("ClaudeText: got %q want %q", got, "ab")
	}
}

func TestParseJSONFromClaude(t *testing.T) {
	t.Parallel()

	type out struct {
		A int `json:"a"`
	}

	cases := []struct {
		name    string
		raw     string
		want    out
		wantErr bool
	}{
		{name: "Empty", raw: "", wantErr: true},
		{name: "MissingObject", raw: "no json here", wantErr: true},
		{name: "Plain", raw: `{"a": 1}`, want: out{A: 1}},
		{name: "Fenced", raw: "```json\n{\"a\": 2}\n```", want: out{A: 2}},
		{name: "Wrapped", raw: "prefix\n{\"a\":3}\nsuffix", want: out{A: 3}},
		{name: "BadJSON", raw: "{bad}", wantErr: true},
		{name: "UnmarshalTypeError", raw: `{"a":"x"}`, wantErr: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			var got out
			err := ParseJSONFromClaude(tc.raw, &got)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ParseJSONFromClaude: expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseJSONFromClaude: %v", err)
			}
			if got != tc.want {
				t.Fatalf("ParseJSONFromClaude: got %#v want %#v", got, tc.want)
			}
		})
	}
}
