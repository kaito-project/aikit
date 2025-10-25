package packager

import (
	"context"
	"strings"
	"testing"

	"github.com/moby/buildkit/client/llb"
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
		{
			name:     "whitespace only",
			input:    "   ",
			expected: nil,
		},
		{
			name:     "unclosed quote",
			input:    "'pattern",
			expected: []string{"pattern"}, // Parser captures content until end
		},
		{
			name:     "empty quotes",
			input:    "''",
			expected: nil, // Empty pattern is not added
		},
		{
			name:     "pattern with spaces inside quotes",
			input:    "'pattern with spaces'",
			expected: []string{"pattern with spaces"},
		},
		{
			name:     "consecutive quotes",
			input:    "''  ''  ''",
			expected: nil,
		},
		{
			name:     "patterns with special characters",
			input:    "'**/*.bin' '*.safetensors' 'model-[0-9]*.gguf'",
			expected: []string{"**/*.bin", "*.safetensors", "model-[0-9]*.gguf"},
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
	tests := []struct {
		name        string
		source      string
		exclude     string
		expectError bool
		errorMsg    string
		mustContain []string
	}{
		{
			name:    "basic huggingface source",
			source:  "huggingface://org/model@rev123",
			exclude: "",
			mustContain: []string{
				"org/model",
				"--revision rev123",
				"hf download",
				"/run/secrets/hf-token",
			},
		},
		{
			name:    "with exclude patterns",
			source:  "huggingface://org/model@rev123",
			exclude: "'original/*' 'metal/*'",
			mustContain: []string{
				"org/model",
				"--revision rev123",
				"--exclude 'original/*' 'metal/*'",
				"hf download",
			},
		},
		{
			name:        "non-huggingface source",
			source:      "https://example.com/model.bin",
			exclude:     "",
			expectError: true,
			errorMsg:    "not a huggingface source",
		},
		{
			name:        "invalid huggingface URL",
			source:      "huggingface://",
			exclude:     "",
			expectError: true,
			errorMsg:    "invalid huggingface source",
		},
		{
			name:        "malformed huggingface path",
			source:      "huggingface://org",
			exclude:     "",
			expectError: true,
			errorMsg:    "invalid huggingface source",
		},
		{
			name:    "valid huggingface source",
			source:  "huggingface://org/model@main",
			exclude: "",
			mustContain: []string{
				"org/model",
				"--revision main",
			},
		},
		{
			name:    "valid with single exclude pattern",
			source:  "huggingface://org/model@v1.0",
			exclude: "'*.bin'",
			mustContain: []string{
				"org/model",
				"--exclude '*.bin'",
			},
		},
		{
			name:    "multiple exclude patterns",
			source:  "huggingface://org/model",
			exclude: "'original/*' 'metal/*' '*.lock'",
			mustContain: []string{
				"org/model",
				"--exclude 'original/*' 'metal/*' '*.lock'",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			st, err := buildHuggingFaceState(tt.source, tt.exclude)
			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.errorMsg)
				}
				if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Fatalf("expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
				return
			}
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

			for _, expect := range tt.mustContain {
				if !strings.Contains(combined, expect) {
					t.Fatalf("expected def to contain %q, got: %s", expect, combined)
				}
			}
		})
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
		st, err := resolveSourceState(cse.src, session, cse.preserve, "")
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

// Test internal helper functions for build configuration parsing.

func Test_parseBuildConfig(t *testing.T) {
	tests := []struct {
		name        string
		opts        map[string]string
		sessionID   string
		isModelpack bool
		expectError bool
		errorMsg    string
		validate    func(*testing.T, *buildConfig)
	}{
		{
			name:        "missing source for modelpack",
			opts:        map[string]string{},
			sessionID:   "session123",
			isModelpack: true,
			expectError: true,
			errorMsg:    "source is required for modelpack target",
		},
		{
			name:        "missing source for generic",
			opts:        map[string]string{},
			sessionID:   "session123",
			isModelpack: false,
			expectError: true,
			errorMsg:    "source is required for generic target",
		},
		{
			name: "empty source string",
			opts: map[string]string{
				"build-arg:source": "",
			},
			sessionID:   "session123",
			isModelpack: true,
			expectError: true,
			errorMsg:    "source is required",
		},
		{
			name: "valid minimal config",
			opts: map[string]string{
				"build-arg:source": "https://example.com/model.bin",
			},
			sessionID:   "session123",
			isModelpack: false,
			expectError: false,
			validate: func(t *testing.T, cfg *buildConfig) {
				if cfg.source != "https://example.com/model.bin" {
					t.Errorf("expected source https://example.com/model.bin, got %s", cfg.source)
				}
				if cfg.packMode != packModeRaw {
					t.Errorf("expected default pack mode %s, got %s", packModeRaw, cfg.packMode)
				}
			},
		},
		{
			name: "custom pack mode",
			opts: map[string]string{
				"build-arg:source":          ".",
				"build-arg:layer_packaging": "tar+gzip",
			},
			sessionID:   "session123",
			isModelpack: false,
			expectError: false,
			validate: func(t *testing.T, cfg *buildConfig) {
				if cfg.packMode != "tar+gzip" {
					t.Errorf("expected pack mode tar+gzip, got %s", cfg.packMode)
				}
			},
		},
		{
			name: "debug flag parsing",
			opts: map[string]string{
				"build-arg:source": ".",
				"build-arg:debug":  "1",
			},
			sessionID:   "session123",
			isModelpack: false,
			expectError: false,
			validate: func(t *testing.T, cfg *buildConfig) {
				if !cfg.debug {
					t.Error("expected debug to be true")
				}
			},
		},
		{
			name: "exclude patterns",
			opts: map[string]string{
				"build-arg:source":  "huggingface://org/model",
				"build-arg:exclude": "'*.bin' '*.safetensors'",
			},
			sessionID:   "session123",
			isModelpack: true,
			expectError: false,
			validate: func(t *testing.T, cfg *buildConfig) {
				if cfg.exclude != "'*.bin' '*.safetensors'" {
					t.Errorf("expected exclude patterns, got %s", cfg.exclude)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseBuildConfig(tt.opts, tt.sessionID, tt.isModelpack)
			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.errorMsg)
				}
				if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Fatalf("expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if cfg == nil {
					t.Fatal("expected non-nil config")
				}
				if tt.validate != nil {
					tt.validate(t, cfg)
				}
			}
		})
	}
}

func Test_determineName(t *testing.T) {
	tests := []struct {
		name     string
		opts     map[string]string
		expected string
	}{
		{
			name:     "nil opts defaults to aikitmodel",
			opts:     nil,
			expected: "aikitmodel",
		},
		{
			name:     "empty opts defaults to aikitmodel",
			opts:     map[string]string{},
			expected: "aikitmodel",
		},
		{
			name: "name provided",
			opts: map[string]string{
				"build-arg:name": "mymodel",
			},
			expected: "mymodel",
		},
		{
			name: "empty name falls back to default",
			opts: map[string]string{
				"build-arg:name": "",
			},
			expected: "aikitmodel",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := determineName(tt.opts)
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func Test_determineRefName(t *testing.T) {
	tests := []struct {
		name     string
		opts     map[string]string
		expected string
	}{
		{
			name:     "nil opts defaults to latest",
			opts:     nil,
			expected: "latest",
		},
		{
			name:     "empty opts defaults to latest",
			opts:     map[string]string{},
			expected: "latest",
		},
		{
			name: "name provided",
			opts: map[string]string{
				"build-arg:name": "v1.0.0",
			},
			expected: "v1.0.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := determineRefName(tt.opts)
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func Test_getBuildArg(t *testing.T) {
	tests := []struct {
		name     string
		opts     map[string]string
		key      string
		expected string
	}{
		{
			name:     "nil opts returns empty",
			opts:     nil,
			key:      "source",
			expected: "",
		},
		{
			name:     "empty opts returns empty",
			opts:     map[string]string{},
			key:      "source",
			expected: "",
		},
		{
			name: "key exists",
			opts: map[string]string{
				"build-arg:source": "value",
			},
			key:      "source",
			expected: "value",
		},
		{
			name: "key does not exist",
			opts: map[string]string{
				"build-arg:other": "value",
			},
			key:      "source",
			expected: "",
		},
		{
			name: "empty value",
			opts: map[string]string{
				"build-arg:source": "",
			},
			key:      "source",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getBuildArg(tt.opts, tt.key)
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// Test_generateHFSingleFileDownloadScript verifies script generation for single-file HF downloads.
func Test_generateHFSingleFileDownloadScript(t *testing.T) {
	tests := []struct {
		name      string
		namespace string
		model     string
		revision  string
		filePath  string
		contains  []string
	}{
		{
			name:      "basic single file",
			namespace: "meta-llama",
			model:     "Llama-3.2-1B",
			revision:  "main",
			filePath:  "config.json",
			contains: []string{
				"set -euo pipefail",
				"hf download meta-llama/Llama-3.2-1B config.json --revision main",
				"mkdir -p /out",
				"--local-dir /out",
				"rm -rf /out/.cache",
			},
		},
		{
			name:      "nested file path",
			namespace: "org",
			model:     "model-name",
			revision:  "v1.0",
			filePath:  "weights/model.safetensors",
			contains: []string{
				"hf download org/model-name weights/model.safetensors --revision v1.0",
			},
		},
		{
			name:      "special characters in revision",
			namespace: "user",
			model:     "repo",
			revision:  "feature/branch-name",
			filePath:  "file.bin",
			contains: []string{
				"hf download user/repo file.bin --revision feature/branch-name",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			script := generateHFSingleFileDownloadScript(tt.namespace, tt.model, tt.revision, tt.filePath)
			for _, substr := range tt.contains {
				if !strings.Contains(script, substr) {
					t.Errorf("expected script to contain %q\nGot script:\n%s", substr, script)
				}
			}
			// Verify script has token handling
			if !strings.Contains(script, "HF_TOKEN") {
				t.Error("expected script to handle HF token")
			}
		})
	}
}

// Test_resolveSourceState_AllPaths tests all code paths in resolveSourceState.
func Test_resolveSourceState_AllPaths(t *testing.T) {
	sessionID := "test-session-123"

	tests := []struct {
		name              string
		source            string
		preserveHTTP      bool
		exclude           string
		expectError       bool
		validateState     func(t *testing.T, st llb.State)
		skipStateValidate bool // For cases where we can't easily validate the state
	}{
		{
			name:         "empty source returns local context",
			source:       "",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, localNameContext) {
					t.Error("expected local context")
				}
			},
		},
		{
			name:         "dot source returns local context",
			source:       ".",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, localNameContext) {
					t.Error("expected local context")
				}
			},
		},
		{
			name:         "context keyword returns local context",
			source:       "context",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, localNameContext) {
					t.Error("expected local context")
				}
			},
		},
		{
			name:         "http URL without preserve filename",
			source:       "http://example.com/model.bin",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, "example.com/model.bin") {
					t.Error("expected HTTP URL in state")
				}
			},
		},
		{
			name:         "https URL with preserve filename",
			source:       "https://example.com/path/model.safetensors",
			preserveHTTP: true,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, "model.safetensors") {
					t.Error("expected preserved filename")
				}
			},
		},
		{
			name:              "huggingface full repo download",
			source:            "huggingface://meta-llama/Llama-3.2-1B@main",
			preserveHTTP:      false,
			exclude:           "*.md *.txt",
			expectError:       false,
			skipStateValidate: true, // HF state is complex, just verify no error
		},
		{
			name:              "huggingface single file download",
			source:            "huggingface://org/model@rev/weights/file.bin",
			preserveHTTP:      false,
			exclude:           "",
			expectError:       false,
			skipStateValidate: true, // Complex multi-step state
		},
		{
			name:         "local path without trailing slash",
			source:       "models/weights",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, "models/weights") {
					t.Error("expected local path pattern")
				}
			},
		},
		{
			name:         "local path with trailing slash",
			source:       "models/",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				// Should append ** for directory globbing
				if !strings.Contains(combined, "models/") {
					t.Error("expected directory path")
				}
			},
		},
		{
			name:         "local glob pattern",
			source:       "*.safetensors",
			preserveHTTP: false,
			exclude:      "",
			expectError:  false,
			validateState: func(t *testing.T, st llb.State) {
				def, _ := st.Marshal(context.Background())
				combined := marshalToString(def)
				if !strings.Contains(combined, "safetensors") {
					t.Error("expected glob pattern")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			st, err := resolveSourceState(tt.source, sessionID, tt.preserveHTTP, tt.exclude)

			if tt.expectError && err == nil {
				t.Fatal("expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if !tt.expectError && !tt.skipStateValidate && tt.validateState != nil {
				tt.validateState(t, st)
			}
		})
	}
}

// marshalToString is a helper to convert LLB state to string for validation.
func marshalToString(def *llb.Definition) string {
	if def == nil {
		return ""
	}
	var combined string
	for _, d := range def.ToPB().Def {
		combined += string(d)
	}
	return combined
}

// Test_BuildModelpack_ConfigValidation tests BuildModelpack configuration parsing
// Note: Full integration testing is done in CI. This tests config validation paths.
func Test_BuildModelpack_ConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		buildOpts   map[string]string
		expectError bool
		errorMsg    string
	}{
		{
			name: "missing source",
			buildOpts: map[string]string{
				"build-arg:name": "test-model",
			},
			expectError: true,
			errorMsg:    "source is required",
		},
		{
			name: "valid minimal config",
			buildOpts: map[string]string{
				"build-arg:source": ".",
				"build-arg:name":   "test-model",
			},
			expectError: false,
		},
		{
			name: "with all options",
			buildOpts: map[string]string{
				"build-arg:source":    "huggingface://org/model@main",
				"build-arg:name":      "my-model",
				"build-arg:ref-name":  "latest",
				"build-arg:pack-mode": "raw",
				"build-arg:exclude":   "*.md",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test parseBuildConfig which is called by BuildModelpack
			_, err := parseBuildConfig(tt.buildOpts, "test-session", true)

			if tt.expectError && err == nil {
				t.Fatal("expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.expectError && err != nil && tt.errorMsg != "" {
				if !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			}
		})
	}
}

// Test_BuildGeneric_ConfigValidation tests BuildGeneric configuration parsing.
func Test_BuildGeneric_ConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		buildOpts   map[string]string
		expectError bool
		errorMsg    string
	}{
		{
			name: "missing source",
			buildOpts: map[string]string{
				"build-arg:name": "test-artifact",
			},
			expectError: true,
			errorMsg:    "source is required",
		},
		{
			name: "valid minimal config",
			buildOpts: map[string]string{
				"build-arg:source": ".",
				"build-arg:name":   "test-artifact",
			},
			expectError: false,
		},
		{
			name: "files output mode",
			buildOpts: map[string]string{
				"build-arg:source": "models/",
				"build-arg:name":   "my-files",
				"build-arg:output": "files",
			},
			expectError: false,
		},
		{
			name: "with debug flag",
			buildOpts: map[string]string{
				"build-arg:source": "https://example.com/data.tar.gz",
				"build-arg:name":   "test",
				"build-arg:debug":  "true",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test parseBuildConfig which is called by BuildGeneric
			_, err := parseBuildConfig(tt.buildOpts, "test-session", false)

			if tt.expectError && err == nil {
				t.Fatal("expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.expectError && err != nil && tt.errorMsg != "" {
				if !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			}
		})
	}
}

// Test_resolveSourceState_ErrorCases tests error handling in resolveSourceState.
func Test_resolveSourceState_ErrorCases(t *testing.T) {
	sessionID := "test-session"

	tests := []struct {
		name        string
		source      string
		exclude     string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "invalid huggingface URL with malformed spec",
			source:      "huggingface://invalid spec format",
			exclude:     "",
			expectError: true, // Will return error for invalid spec
			errorMsg:    "invalid huggingface",
		},
		{
			name:        "huggingface repo with exclude pattern",
			source:      "huggingface://org/model@main",
			exclude:     "*.txt *.md",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := resolveSourceState(tt.source, sessionID, false, tt.exclude)

			if tt.expectError && err == nil {
				t.Fatal("expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.expectError && err != nil && tt.errorMsg != "" {
				if !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			}
		})
	}
}
