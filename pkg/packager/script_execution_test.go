package packager

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type testLayoutDescriptor struct {
	Digest      string            `json:"digest"`
	Size        int64             `json:"size"`
	Annotations map[string]string `json:"annotations"`
}

type testLayoutIndex struct {
	Manifests []testLayoutDescriptor `json:"manifests"`
}

type testLayoutManifest struct {
	Layers []testLayoutDescriptor `json:"layers"`
}

func TestEmbeddedPackagingScriptsPreserveSpecialFilenames(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("embedded packager scripts use Linux coreutils/BusyBox interfaces")
	}
	for _, command := range []string{"bash", "find", "sort", "xargs", "stat", "sha256sum", "tar", "sed", "awk"} {
		if _, err := exec.LookPath(command); err != nil {
			t.Skipf("%s is required to execute embedded packager scripts: %v", command, err)
		}
	}

	files := map[string][]byte{
		"plain.gguf":        []byte("plain"),
		"pipe|name.txt":     []byte("pipe"),
		"line\nbreak.gguf":  []byte("newline"),
		"tab\tname.json":    []byte("tab"),
		"space name.md":     []byte("space"),
		"-leading-dash.bin": []byte("dash"),
		"nested/a|b\nc.py":  []byte("nested"),
	}

	tests := []struct {
		name      string
		script    string
		modelpack bool
	}{
		{name: "generic", script: genericScript},
		{name: "modelpack", script: modelpackScript, modelpack: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sourceDir := t.TempDir()
			layoutDir := t.TempDir()
			tmpDir := t.TempDir()
			for name, content := range files {
				path := filepath.Join(sourceDir, filepath.FromSlash(name))
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatalf("create parent directory for %q: %v", name, err)
				}
				if err := os.WriteFile(path, content, 0o600); err != nil {
					t.Fatalf("write source file %q: %v", name, err)
				}
			}

			cmd := exec.Command("bash", "-c", tt.script) //nolint:gosec // Embedded repository script, not user input.
			cmd.Env = append(os.Environ(),
				"AIKIT_SOURCE_DIR="+sourceDir,
				"AIKIT_LAYOUT_DIR="+layoutDir,
				"AIKIT_TMP_DIR="+tmpDir,
				"PACK_MODE=raw",
				"ARTIFACT_TYPE=application/vnd.test.artifact",
				"NAME=test-artifact",
				"REF_NAME=latest",
			)
			if tt.modelpack {
				cmd.Env = append(cmd.Env,
					"MT_MANIFEST=application/vnd.test.config",
					"LARGE_FILE_THRESHOLD=10485760",
				)
			} else {
				cmd.Env = append(cmd.Env,
					"RAW_LAYER_MT=application/octet-stream",
					"ARCHIVE_LAYER_MT=application/vnd.oci.image.layer.v1.tar",
				)
			}
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("execute %s script: %v\n%s", tt.name, err, output)
			}

			assertLayoutSpecialFilenames(t, layoutDir, files, tt.modelpack)
		})
	}
}

func assertLayoutSpecialFilenames(t *testing.T, layoutDir string, files map[string][]byte, modelpack bool) {
	t.Helper()

	var index testLayoutIndex
	readLayoutJSON(t, filepath.Join(layoutDir, "index.json"), &index)
	if len(index.Manifests) != 1 {
		t.Fatalf("manifest descriptors = %d, want 1", len(index.Manifests))
	}

	var manifest testLayoutManifest
	readLayoutJSON(t, layoutBlobPath(t, layoutDir, index.Manifests[0].Digest), &manifest)
	if len(manifest.Layers) != len(files) {
		t.Fatalf("layers = %d, want %d", len(manifest.Layers), len(files))
	}

	seen := make(map[string]bool, len(files))
	for _, layer := range manifest.Layers {
		title := layer.Annotations["org.opencontainers.image.title"]
		wantContent, ok := files[title]
		if !ok {
			t.Fatalf("unexpected layer title %q", title)
		}
		if seen[title] {
			t.Fatalf("duplicate layer title %q", title)
		}
		seen[title] = true
		if layer.Size != int64(len(wantContent)) {
			t.Errorf("layer %q size = %d, want %d", title, layer.Size, len(wantContent))
		}
		gotContent, err := os.ReadFile(layoutBlobPath(t, layoutDir, layer.Digest))
		if err != nil {
			t.Fatalf("read layer blob for %q: %v", title, err)
		}
		if string(gotContent) != string(wantContent) {
			t.Errorf("layer %q content = %q, want %q", title, gotContent, wantContent)
		}
		if modelpack {
			if got := layer.Annotations["org.cncf.model.filepath"]; got != title {
				t.Errorf("modelpack filepath annotation = %q, want %q", got, title)
			}
			var metadata struct {
				Name string `json:"name"`
			}
			if err := json.Unmarshal([]byte(layer.Annotations["org.cncf.model.file.metadata+json"]), &metadata); err != nil {
				t.Fatalf("decode modelpack metadata for %q: %v", title, err)
			}
			if metadata.Name != title {
				t.Errorf("modelpack metadata name = %q, want %q", metadata.Name, title)
			}
		}
	}
	for name := range files {
		if !seen[name] {
			t.Errorf("missing layer for %q", name)
		}
	}
}

func readLayoutJSON(t *testing.T, path string, target any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		t.Fatalf("decode %s: %v\n%s", path, err, data)
	}
}

func layoutBlobPath(t *testing.T, layoutDir, digest string) string {
	t.Helper()
	algorithm, encoded, ok := strings.Cut(digest, ":")
	if !ok || algorithm != "sha256" || encoded == "" {
		t.Fatalf("invalid layout digest %q", digest)
	}
	return filepath.Join(layoutDir, "blobs", algorithm, encoded)
}
