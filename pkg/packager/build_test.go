package packager

import (
	"context"
	"strings"
	"testing"
)

func Test_generateHFDownloadScript(t *testing.T) {
	script := generateHFDownloadScript("org", "model", "rev123", "")
	checks := []string{
		"set -euo pipefail",
		"org/model",
		"--revision rev123",
		"/run/secrets/hf-token",
		"hf download",
		"rm -rf /out/.cache",
		"find /out -type f -name '*.lock' -delete || true",
	}
	for _, c := range checks {
		if !strings.Contains(script, c) {
			t.Fatalf("expected script to contain %q; got %s", c, script)
		}
	}
	// Ensure no accidental printf tokens remain unexpanded
	if strings.Contains(script, "%s") {
		t.Fatalf("unexpected unexpanded fmt token in script: %s", script)
	}
}

func Test_generateHFDownloadScript_WithExclude(t *testing.T) {
	script := generateHFDownloadScript("org", "model", "rev123", "'original/*' 'metal/*'")
	checks := []string{
		"set -euo pipefail",
		"org/model",
		"--revision rev123",
		"--exclude 'original/*' 'metal/*'",
		"hf download",
	}
	for _, c := range checks {
		if !strings.Contains(script, c) {
			t.Fatalf("expected script to contain %q; got %s", c, script)
		}
	}
}

func Test_parseExcludePatterns(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: nil,
		},
		{
			name:     "single quoted pattern",
			input:    "'original/*'",
			expected: []string{"original/*"},
		},
		{
			name:     "multiple quoted patterns",
			input:    "'original/*' 'metal/*'",
			expected: []string{"original/*", "metal/*"},
		},
		{
			name:     "double quotes",
			input:    `"*.safetensors" "metal/**"`,
			expected: []string{"*.safetensors", "metal/**"},
		},
		{
			name:     "mixed patterns",
			input:    "'original/**' \"metal/*\" '*.bin'",
			expected: []string{"original/**", "metal/*", "*.bin"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseExcludePatterns(tt.input)
			if len(result) != len(tt.expected) {
				t.Fatalf("expected %d patterns, got %d: %v", len(tt.expected), len(result), result)
			}
			for i, exp := range tt.expected {
				if result[i] != exp {
					t.Fatalf("pattern %d: expected %q, got %q", i, exp, result[i])
				}
			}
		})
	}
}

func Test_createMinimalImageConfig(t *testing.T) {
	b, err := createMinimalImageConfig("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	s := string(b)
	expect := []string{"\"os\":\"linux\"", "\"architecture\":\"amd64\"", "\"rootfs\""}
	for _, e := range expect {
		if !strings.Contains(s, e) {
			t.Fatalf("expected config JSON to contain %s, got %s", e, s)
		}
	}
	if !strings.Contains(s, "layers") {
		t.Fatalf("expected empty layers rootfs, got %s", s)
	}
}

func Test_buildHuggingFaceState_ScriptContent(t *testing.T) {
	src := "huggingface://org/model@rev123"
	st, err := buildHuggingFaceState(src, "", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	def, err := st.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var combined string
	for _, d := range def.ToPB().Def {
		combined += string(d)
	}
	for _, expect := range []string{"org/model", "--revision rev123", "hf download"} {
		if !strings.Contains(combined, expect) {
			t.Fatalf("expected def to contain %q", expect)
		}
	}
	// Secret mount now unconditional; ensure presence even with empty flag.
	if !strings.Contains(combined, "/run/secrets/hf-token") {
		t.Fatalf("expected secret mount directive in definition: %s", combined)
	}
}

func Test_buildHuggingFaceState_SecretMount(t *testing.T) {
	src := "huggingface://org/model@main"
	st, err := buildHuggingFaceState(src, "", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	def, err := st.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var combined string
	for _, d := range def.ToPB().Def {
		combined += string(d)
	}
	if !strings.Contains(combined, "/run/secrets/hf-token") {
		t.Fatalf("expected secret mount path in definition")
	}
}

func Test_buildHuggingFaceState_WithExclude(t *testing.T) {
	src := "huggingface://org/model@rev123"
	exclude := "'original/*' 'metal/*'"
	st, err := buildHuggingFaceState(src, "", exclude)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	def, err := st.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var combined string
	for _, d := range def.ToPB().Def {
		combined += string(d)
	}
	// Check that exclude patterns are in the command as a single --exclude flag
	if !strings.Contains(combined, "--exclude 'original/*' 'metal/*'") {
		t.Fatalf("expected def to contain \"--exclude 'original/*' 'metal/*'\", got: %s", combined)
	}
}

func Test_resolveSourceState_Variants(t *testing.T) {
	session := "sess123"
	cases := []struct {
		src      string
		preserve bool
		expect   string
	}{
		{"context", true, localNameContext},
		{".", false, localNameContext},
		{"https://example.com/file.bin", true, "file.bin"},
		{"https://example.com/file.bin", false, "file.bin"},
		{"huggingface://org/model@rev", false, "hf download"},
		{"subdir/", false, "subdir"},
	}
	for _, cse := range cases {
		st, err := resolveSourceState(cse.src, session, "", cse.preserve, "")
		if err != nil {
			t.Fatalf("resolve failed for %s: %v", cse.src, err)
		}
		def, err := st.Marshal(context.Background())
		if err != nil {
			t.Fatalf("marshal failed: %v", err)
		}
		var combined string
		for _, d := range def.ToPB().Def {
			combined += string(d)
		}
		if !strings.Contains(combined, cse.expect) {
			t.Fatalf("expected def for %s to contain %q (got %s)", cse.src, cse.expect, combined)
		}
	}
}

func Test_generateModelpackScript(t *testing.T) {
	script := generateModelpackScript("raw", "art.type", "mt.conf", "myname", "refy")
	mustContain := []string{
		"PACK_MODE=raw",
		"art.type",
		"mt.conf",
		"org.opencontainers.image.title\": \"myname\"",
		"org.opencontainers.image.ref.name\": \"refy\"",
		"add_category /tmp/weights.list weights",
	}
	for _, s := range mustContain {
		if !strings.Contains(script, s) {
			t.Fatalf("expected script to contain %q", s)
		}
	}
}

func Test_generateGenericScript(t *testing.T) {
	script := generateGenericScript("tar+gzip", "atype", "nm", "refz", true)
	checks := []string{
		"set -x",
		"PACK_MODE=tar+gzip",
		"atype",
		"org.opencontainers.image.title\": \"nm\"",
		"org.opencontainers.image.ref.name\": \"refz\"",
	}
	for _, c := range checks {
		if !strings.Contains(script, c) {
			t.Fatalf("missing %q in generic script", c)
		}
	}
}

func Test_generateGenericScript_RawOctetStream(t *testing.T) {
	script := generateGenericScript("raw", "atype2", "nm2", "ref2", false)
	if !strings.Contains(script, "application/octet-stream") {
		t.Fatalf("expected raw generic script to use application/octet-stream media type, got: %s", script)
	}
	if !strings.Contains(script, "PACK_MODE=raw") {
		t.Fatalf("expected PACK_MODE=raw in script")
	}
}
