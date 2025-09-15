package packager

import (
	"encoding/json"
	"fmt"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// Shared container image references.
const (
	bashImage  = "cgr.dev/chainguard/bash:latest"
	hfCLIImage = "ghcr.io/kaito-project/aikit/hf-cli:latest"
)

// generateHFDownloadScript returns a shell script that downloads a Hugging Face
// repository snapshot deterministically, honoring an optional token exposed
// through a BuildKit secret at /run/secrets/hf-token.
func generateHFDownloadScript(namespace, model, revision string) string {
	return fmt.Sprintf(`set -euo pipefail
if [ -f /run/secrets/hf-token ]; then export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf-token)"; fi
mkdir -p /out
hf download %s/%s --revision %s --local-dir /out
# remove transient cache / lock artifacts
rm -rf /out/.cache || true
find /out -type f -name '*.lock' -delete || true
`, namespace, model, revision)
}

// generateHFSingleFileDownloadScript downloads a single file from a Hugging Face
// repository deterministically. filePath is the relative path inside the repo.
func generateHFSingleFileDownloadScript(namespace, model, revision, filePath string) string {
	return fmt.Sprintf(`set -euo pipefail
if [ -f /run/secrets/hf-token ]; then export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf-token)"; fi
mkdir -p /out
hf download %s/%s %s --revision %s --local-dir /out
# remove transient cache / lock artifacts
rm -rf /out/.cache || true
find /out -type f -name '*.lock' -delete || true
`, namespace, model, filePath, revision)
}

// createMinimalImageConfig produces a serialized minimal OCI image config JSON
// with provided OS and architecture. RootFS is empty (no layers) matching other
// packager outputs.
func createMinimalImageConfig(os, arch string) ([]byte, error) {
	cfg := ocispec.Image{}
	cfg.OS = os
	cfg.Architecture = arch
	cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
	return json.Marshal(cfg)
}
