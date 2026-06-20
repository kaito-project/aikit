package packager

import (
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

// extractEscapeJSON pulls the escape_json shell function definition out of the
// embedded script so the test exercises the EXACT escaping logic shipped in the
// container, rather than a reimplementation that could drift from it.
func extractEscapeJSON(t *testing.T, script string) string {
	t.Helper()
	start := strings.Index(script, "escape_json() {")
	if start < 0 {
		t.Fatal("escape_json function not found in embedded script")
	}
	rest := script[start:]
	end := strings.Index(rest, "\n}")
	if end < 0 {
		t.Fatal("could not find end of escape_json function")
	}
	return rest[:end+2]
}

// runEscapeJSON runs the embedded escape_json function against the given input
// and returns its output.
func runEscapeJSON(t *testing.T, fn, input string) string {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}
	script := fn + "\nescape_json \"$1\"\n"
	cmd := exec.Command("bash", "-s", "--", input)
	cmd.Stdin = strings.NewReader(script)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("running escape_json: %v", err)
	}
	return string(out)
}

// TestEscapeJSONProducesValidJSON is the regression test for the verified bug:
// the previous annotation builder used `sed 's/"/\\"/g'` which only escaped
// double-quotes (and interpolated values with no escaping at all), so a file
// path containing a quote, backslash, or other special character produced
// invalid JSON. The fixed escape_json must yield a value that embeds cleanly.
func TestEscapeJSONProducesValidJSON(t *testing.T) {
	for _, script := range []string{modelpackScript, genericScript} {
		fn := extractEscapeJSON(t, script)
		inputs := []string{
			`plain.gguf`,
			`weird"name.gguf`,
			`back\slash.bin`,
			`both"and\here.safetensors`,
			`tab	and spaces.txt`,
		}
		for _, in := range inputs {
			escaped := runEscapeJSON(t, fn, in)
			doc := `{ "title": "` + escaped + `" }`
			var parsed struct {
				Title string `json:"title"`
			}
			if err := json.Unmarshal([]byte(doc), &parsed); err != nil {
				t.Errorf("input %q produced invalid JSON %q: %v", in, doc, err)
				continue
			}
			if parsed.Title != in {
				t.Errorf("round-trip mismatch: input %q, got %q", in, parsed.Title)
			}
		}
	}
}
