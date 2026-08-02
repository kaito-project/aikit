package packager

import (
	"strings"
	"testing"
)

const (
	performanceScriptModelpack = "modelpack"
	performanceScriptGeneric   = "generic"
)

func TestEmbeddedPackagingScriptsStreamLayerJSON(t *testing.T) {
	tests := []struct {
		name   string
		script string
	}{
		{
			name:   performanceScriptModelpack,
			script: modelpackScript,
		},
		{
			name:   performanceScriptGeneric,
			script: genericScript,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mustContain := []string{
				`layers_file="$tmp_dir/layers.json"`,
				`: > "$layers_file"`,
				"first_layer=1",
				`printf ' , ' >> "$layers_file"`,
				`cat "$layers_file"`,
			}
			for _, pattern := range mustContain {
				if !strings.Contains(tt.script, pattern) {
					t.Fatalf("expected embedded script to contain %q", pattern)
				}
			}

			if strings.Contains(tt.script, "layers_json") {
				t.Fatal("embedded script must not build layer JSON through repeated shell string concatenation")
			}
			if count := strings.Count(tt.script, `cat "$layers_file"`); count != 1 {
				t.Fatalf("expected embedded script to read streamed layer JSON once, got %d reads", count)
			}
		})
	}
}

func TestEmbeddedPackagingScriptsAvoidRepeatedPerFileScans(t *testing.T) {
	tests := []struct {
		name   string
		script string
	}{
		{
			name:   performanceScriptModelpack,
			script: modelpackScript,
		},
		{
			name:   performanceScriptGeneric,
			script: genericScript,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mustContain := []string{
				`-print0 | LC_ALL=C sort -z`,
				`xargs -0 -r stat -c '%s' --`,
				`printf '%s\0%s\0'`,
				`read -r -d ''`,
				`read -r dgst _ < <(sha256sum < "$file")`,
			}
			for _, pattern := range mustContain {
				if !strings.Contains(tt.script, pattern) {
					t.Fatalf("expected embedded script to contain %q", pattern)
				}
			}

			mustNotContain := []string{
				"get_cached_size",
				"get_file_size",
				"grep -F",
				"cut -d'|'",
				"head -n1",
				`IFS='|'`,
				`'%n|%s'`,
				`basename "$f"`,
				`sha256sum "$file" |`,
			}
			for _, pattern := range mustNotContain {
				if strings.Contains(tt.script, pattern) {
					t.Fatalf("embedded script must not contain repeated-scan pattern %q", pattern)
				}
			}
		})
	}
}

func TestEmbeddedPackagingScriptsBoundArchiveArgumentSize(t *testing.T) {
	tests := []struct {
		name   string
		script string
	}{
		{name: performanceScriptModelpack, script: modelpackScript},
		{name: performanceScriptGeneric, script: genericScript},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mustContain := []string{
				"archive_batch_bytes=32768",
				`xargs -0 -r -s "$archive_batch_bytes" bash "$tar_batch_script"`,
				`tar -cf "$chunk" --no-recursion -- "$@" .`,
				`cat "$zero_block" "$zero_block" >> "$archive"`,
			}
			for _, pattern := range mustContain {
				if !strings.Contains(tt.script, pattern) {
					t.Fatalf("expected embedded script to contain bounded archive pattern %q", pattern)
				}
			}

			for _, pattern := range []string{`"${files[@]}"`, "local -a files"} {
				if strings.Contains(tt.script, pattern) {
					t.Fatalf("embedded script must not expand an unbounded archive member array %q", pattern)
				}
			}
		})
	}
}

func TestEmbeddedRawPackagingScriptsCopyDirectlyToDigestBlob(t *testing.T) {
	if !strings.Contains(modelpackScript, `append_layer "$f" "$mtRaw" "$f" "$meta" true "$fsize" copy`) {
		t.Fatal("expected modelpack raw mode to copy source files directly into digest-addressed blobs")
	}

	if !strings.Contains(genericScript, `append_layer "$f" "$RAW_LAYER_MT" "$f" "$fsize" copy`) {
		t.Fatal("expected generic raw mode to copy source files directly into digest-addressed blobs")
	}

	for name, script := range map[string]string{
		performanceScriptModelpack: modelpackScript,
		performanceScriptGeneric:   genericScript,
	} {
		t.Run(name, func(t *testing.T) {
			for _, pattern := range []string{"/tmp/raw-", `"/tmp/$(basename "$f")"`} {
				if strings.Contains(script, pattern) {
					t.Fatalf("embedded raw script must not use basename-derived temporary path %q", pattern)
				}
			}
		})
	}
}

func TestEmbeddedModelpackScriptKeepsDeterministicCategoryOrder(t *testing.T) {
	categories := []string{
		`add_category "$weights_list" weights`,
		`add_category "$config_list" config`,
		`add_category "$docs_list" docs`,
		`add_category "$code_list" code`,
		`add_category "$dataset_list" dataset`,
	}

	previous := -1
	for _, category := range categories {
		index := strings.Index(modelpackScript, category)
		if index < 0 {
			t.Fatalf("expected embedded modelpack script to contain %q", category)
		}
		if index <= previous {
			t.Fatalf("expected %q after the previous category", category)
		}
		previous = index
	}
}

func TestEmbeddedModelpackScriptUsesCompleteParentDirectory(t *testing.T) {
	if !strings.Contains(modelpackScript, `dir=${f%/*}`) {
		t.Fatal("expected weight archive generation to use the complete parent directory")
	}
}
