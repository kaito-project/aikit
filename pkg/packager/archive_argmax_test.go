package packager

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const archiveArgumentLimitBytes = 32768

func TestEmbeddedPackagingScriptsArchiveBeyondArgumentLimitInProductionImage(t *testing.T) {
	docker, err := exec.LookPath("docker")
	if err != nil {
		t.Skipf("docker is required for the production-image archive regression test: %v", err)
	}
	if output, inspectErr := exec.Command(docker, "image", "inspect", bashImage).CombinedOutput(); inspectErr != nil {
		t.Skipf("production image %s is not available locally: %v\n%s", bashImage, inspectErr, output)
	}

	sourceDir := t.TempDir()
	files := make(map[string][]byte)
	longDir := strings.Repeat("directory-", 12)
	for i := range 320 {
		name := filepath.ToSlash(filepath.Join(longDir, fmt.Sprintf("artifact-%04d-%060d.txt", i, 0)))
		files[name] = []byte(fmt.Sprintf("payload-%04d", i))
	}
	files["line\nbreak.txt"] = []byte("newline")
	files["trailing-newline.txt\n"] = []byte("trailing newline")
	files["-leading-dash.txt"] = []byte("dash")
	files["zero-blocks.txt"] = make([]byte, 3*512)

	pathBytes := 0
	for name, content := range files {
		pathBytes += len(name) + 1
		path := filepath.Join(sourceDir, filepath.FromSlash(name))
		if mkdirErr := os.MkdirAll(filepath.Dir(path), 0o755); mkdirErr != nil {
			t.Fatalf("create parent directory for %q: %v", name, mkdirErr)
		}
		if writeErr := os.WriteFile(path, content, 0o600); writeErr != nil {
			t.Fatalf("write source file %q: %v", name, writeErr)
		}
	}
	if pathBytes <= archiveArgumentLimitBytes {
		t.Fatalf("test paths use %d bytes, want more than the %d-byte archive argument limit", pathBytes, archiveArgumentLimitBytes)
	}

	wrapperDir := t.TempDir()
	wrapperPath := filepath.Join(wrapperDir, "tar")
	wrapper := `#!/usr/bin/env bash
set -euo pipefail
bytes=0
for arg in "$@"; do
	bytes=$((bytes + ${#arg} + 1))
done
if [ "$bytes" -gt "$AIKIT_TEST_TAR_ARG_LIMIT" ]; then
	printf 'tar argv uses %d bytes, limit is %s\n' "$bytes" "$AIKIT_TEST_TAR_ARG_LIMIT" >&2
	exit 97
fi
printf '%s\n' "$bytes" >> "$AIKIT_TEST_TAR_CALLS"
exec /usr/bin/tar "$@"
`
	if err := os.WriteFile(wrapperPath, []byte(wrapper), 0o755); err != nil { //nolint:gosec // Test-owned executable wrapper.
		t.Fatalf("write tar argument-limit wrapper: %v", err)
	}

	tests := []struct {
		name      string
		script    string
		modelpack bool
	}{
		{name: performanceScriptGeneric, script: genericScript},
		{name: performanceScriptModelpack + "-non-weight", script: modelpackScript, modelpack: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			firstDigest := runProductionArchiveScript(t, docker, sourceDir, wrapperDir, tt.script, tt.modelpack, files)
			secondDigest := runProductionArchiveScript(t, docker, sourceDir, wrapperDir, tt.script, tt.modelpack, files)
			if firstDigest != secondDigest {
				t.Fatalf("archive digest is not deterministic: first %s, second %s", firstDigest, secondDigest)
			}
		})
	}
}

func runProductionArchiveScript(t *testing.T, docker, sourceDir, wrapperDir, script string, modelpack bool, files map[string][]byte) string {
	t.Helper()

	layoutDir := t.TempDir()
	tmpDir := t.TempDir()
	scriptPath := filepath.Join(t.TempDir(), "packager.sh")
	if err := os.WriteFile(scriptPath, []byte(script), 0o600); err != nil {
		t.Fatalf("write embedded packager script: %v", err)
	}

	uid := strings.TrimSpace(commandOutput(t, "id", "-u"))
	gid := strings.TrimSpace(commandOutput(t, "id", "-g"))
	args := []string{
		"run", "--rm", "--user", uid + ":" + gid,
		"--entrypoint", "/bin/bash",
		"-v", sourceDir + ":/src:ro",
		"-v", layoutDir + ":/layout",
		"-v", tmpDir + ":/work",
		"-v", wrapperDir + ":/test-bin:ro",
		"-v", scriptPath + ":/packager.sh:ro",
		"-e", "PATH=/test-bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/bin",
		"-e", "AIKIT_SOURCE_DIR=/src",
		"-e", "AIKIT_LAYOUT_DIR=/layout",
		"-e", "AIKIT_TMP_DIR=/work",
		"-e", "AIKIT_TEST_TAR_ARG_LIMIT=" + strconv.Itoa(archiveArgumentLimitBytes),
		"-e", "AIKIT_TEST_TAR_CALLS=/work/tar-calls",
		"-e", "PACK_MODE=tar",
		"-e", "ARTIFACT_TYPE=application/vnd.test.artifact",
		"-e", "NAME=argument-limit-test",
		"-e", "REF_NAME=latest",
	}
	if modelpack {
		args = append(args,
			"-e", "MT_MANIFEST=application/vnd.test.config",
			"-e", "LARGE_FILE_THRESHOLD=10485760",
		)
	} else {
		args = append(args,
			"-e", "RAW_LAYER_MT=application/octet-stream",
			"-e", "ARCHIVE_LAYER_MT=application/vnd.oci.image.layer.v1.tar",
		)
	}
	args = append(args, bashImage, "/packager.sh")
	cmd := exec.Command(docker, args...) //nolint:gosec // Fixed production image and test-owned paths.
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("execute embedded script in %s: %v\n%s", bashImage, err, output)
	}

	callData, err := os.ReadFile(filepath.Join(tmpDir, "tar-calls"))
	if err != nil {
		t.Fatalf("read tar invocation sizes: %v", err)
	}
	calls := strings.Fields(string(callData))
	if len(calls) < 2 {
		t.Fatalf("tar was invoked %d time(s), want multiple bounded batches", len(calls))
	}
	for _, call := range calls {
		bytesUsed, parseErr := strconv.Atoi(call)
		if parseErr != nil {
			t.Fatalf("parse tar invocation size %q: %v", call, parseErr)
		}
		if bytesUsed > archiveArgumentLimitBytes {
			t.Fatalf("tar invocation used %d bytes, limit is %d", bytesUsed, archiveArgumentLimitBytes)
		}
	}

	var index testLayoutIndex
	readLayoutJSON(t, filepath.Join(layoutDir, "index.json"), &index)
	if len(index.Manifests) != 1 {
		t.Fatalf("manifest descriptors = %d, want 1", len(index.Manifests))
	}
	var manifest testLayoutManifest
	readLayoutJSON(t, layoutBlobPath(t, layoutDir, index.Manifests[0].Digest), &manifest)
	if len(manifest.Layers) != 1 {
		t.Fatalf("archive layers = %d, want 1", len(manifest.Layers))
	}

	blobPath := layoutBlobPath(t, layoutDir, manifest.Layers[0].Digest)
	blob, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatalf("read archive layer: %v", err)
	}
	assertTarMembers(t, blob, files)
	return manifest.Layers[0].Digest
}

func assertTarMembers(t *testing.T, archive []byte, files map[string][]byte) {
	t.Helper()

	reader := tar.NewReader(bytes.NewReader(archive))
	gotOrder := make([]string, 0, len(files))
	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read archive member: %v", err)
		}
		wantContent, ok := files[header.Name]
		if !ok {
			t.Fatalf("unexpected archive member %q", header.Name)
		}
		gotContent, readErr := io.ReadAll(reader)
		if readErr != nil {
			t.Fatalf("read archive member %q: %v", header.Name, readErr)
		}
		if !bytes.Equal(gotContent, wantContent) {
			t.Fatalf("archive member %q content mismatch", header.Name)
		}
		gotOrder = append(gotOrder, header.Name)
	}
	if len(gotOrder) != len(files) {
		t.Fatalf("archive members = %d, want %d", len(gotOrder), len(files))
	}

	wantOrder := make([]string, 0, len(files))
	for name := range files {
		wantOrder = append(wantOrder, name)
	}
	sort.Strings(wantOrder)
	for i := range wantOrder {
		if gotOrder[i] != wantOrder[i] {
			t.Fatalf("archive member %d = %q, want deterministic order %q", i, gotOrder[i], wantOrder[i])
		}
	}
}

func commandOutput(t *testing.T, name string, args ...string) string {
	t.Helper()
	output, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		t.Fatalf("run %s: %v\n%s", name, err, output)
	}
	return string(output)
}
