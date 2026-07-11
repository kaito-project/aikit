package packager

import (
	"strings"
	"testing"
)

const (
	performanceScriptModelpack = "modelpack"
	performanceScriptGeneric   = "generic"
)

func TestGeneratedPackagingScriptsStreamLayerJSON(t *testing.T) {
	tests := []struct {
		name   string
		script string
	}{
		{
			name:   performanceScriptModelpack,
			script: generateModelpackScript(packModeRaw, "art.type", "mt.conf", "model", "latest"),
		},
		{
			name:   performanceScriptGeneric,
			script: generateGenericScript(packModeRaw, "art.type", "artifact", "latest", false),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mustContain := []string{
				"layers_file=/tmp/layers.json",
				`: > "$layers_file"`,
				"first_layer=1",
				`printf ' , ' >> "$layers_file"`,
				`cat "$layers_file"`,
			}
			for _, pattern := range mustContain {
				if !strings.Contains(tt.script, pattern) {
					t.Fatalf("expected generated script to contain %q", pattern)
				}
			}

			if strings.Contains(tt.script, "layers_json") {
				t.Fatal("generated script must not build layer JSON through repeated shell string concatenation")
			}
			if count := strings.Count(tt.script, `cat "$layers_file"`); count != 1 {
				t.Fatalf("expected generated script to read streamed layer JSON once, got %d reads", count)
			}
		})
	}
}

func TestGeneratedPackagingScriptsAvoidRepeatedPerFileScans(t *testing.T) {
	tests := []struct {
		name   string
		script string
	}{
		{
			name:   performanceScriptModelpack,
			script: generateModelpackScript(packModeRaw, "art.type", "mt.conf", "model", "latest"),
		},
		{
			name:   performanceScriptGeneric,
			script: generateGenericScript(packModeRaw, "art.type", "artifact", "latest", false),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mustContain := []string{
				`-exec stat -c '%n|%s' {} +`,
				"declare -A file_sizes=()",
				`file_sizes["$f"]=$sz`,
				`read -r dgst _ < <(sha256sum "$file")`,
				"LC_ALL=C sort",
			}
			for _, pattern := range mustContain {
				if !strings.Contains(tt.script, pattern) {
					t.Fatalf("expected generated script to contain %q", pattern)
				}
			}

			mustNotContain := []string{
				"xargs -0",
				"get_cached_size",
				"get_file_size",
				"grep -F",
				"cut -d'|'",
				"head -n1",
				`basename "$f"`,
				`sha256sum "$file" |`,
			}
			for _, pattern := range mustNotContain {
				if strings.Contains(tt.script, pattern) {
					t.Fatalf("generated script must not contain repeated-scan pattern %q", pattern)
				}
			}
		})
	}
}

func TestGeneratedRawPackagingScriptsCopyDirectlyToDigestBlob(t *testing.T) {
	modelpackScript := generateModelpackScript(packModeRaw, "art.type", "mt.conf", "model", "latest")
	if !strings.Contains(modelpackScript, `append_layer "$f" "$mtRaw" "$f" "$meta" "true" "$fsize" copy`) {
		t.Fatal("expected modelpack raw mode to copy source files directly into digest-addressed blobs")
	}

	genericScript := generateGenericScript(packModeRaw, "art.type", "artifact", "latest", false)
	if !strings.Contains(genericScript, `append_layer "$f" "application/octet-stream" "$f" "${file_sizes["$f"]}" copy`) {
		t.Fatal("expected generic raw mode to copy source files directly into digest-addressed blobs")
	}

	for name, script := range map[string]string{
		performanceScriptModelpack: modelpackScript,
		performanceScriptGeneric:   genericScript,
	} {
		t.Run(name, func(t *testing.T) {
			for _, pattern := range []string{"/tmp/raw-", `"/tmp/$(basename "$f")"`} {
				if strings.Contains(script, pattern) {
					t.Fatalf("generated raw script must not use basename-derived temporary path %q", pattern)
				}
			}
		})
	}
}

func TestGeneratedModelpackScriptKeepsDeterministicCategoryOrder(t *testing.T) {
	script := generateModelpackScript(packModeRaw, "art.type", "mt.conf", "model", "latest")
	categories := []string{
		"add_category /tmp/weights.list weights",
		"add_category /tmp/config.list config",
		"add_category /tmp/docs.list docs",
		"add_category /tmp/code.list code",
		"add_category /tmp/dataset.list dataset",
	}

	previous := -1
	for _, category := range categories {
		index := strings.Index(script, category)
		if index < 0 {
			t.Fatalf("expected generated script to contain %q", category)
		}
		if index <= previous {
			t.Fatalf("expected %q after the previous category", category)
		}
		previous = index
	}
}

func TestGeneratedModelpackScriptUsesCompleteParentDirectory(t *testing.T) {
	script := generateModelpackScript("tar", "art.type", "mt.conf", "model", "latest")
	if !strings.Contains(script, `dir=${f%/*}`) {
		t.Fatal("expected weight archive generation to use the complete parent directory")
	}
}
