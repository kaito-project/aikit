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
// exclude is an optional space-separated list of patterns (e.g., "'original/*' 'metal/*'")
// which will be passed as separate --exclude flags to the hf download command.
func generateHFDownloadScript(namespace, model, revision, exclude string) string {
	excludeFlags := ""
	if exclude != "" {
		// Parse the exclude patterns: they come in as "'pattern1' 'pattern2'"
		// We need to convert this to: --exclude 'pattern1' --exclude 'pattern2'
		// Each pattern requires its own --exclude flag per hf cli syntax
		patterns := parseExcludePatterns(exclude)
		for _, pattern := range patterns {
			excludeFlags += fmt.Sprintf(" --exclude '%s'", pattern)
		}
	}
	return fmt.Sprintf(`set -euo pipefail
if [ -f /run/secrets/hf-token ]; then export HF_TOKEN="$(cat /run/secrets/hf-token)"; fi
mkdir -p /out
hf download %s/%s --revision %s --local-dir /out%s
# remove transient cache / lock artifacts
rm -rf /out/.cache || true
find /out -type f -name '*.lock' -delete || true
`, namespace, model, revision, excludeFlags)
}

// parseExcludePatterns takes a string like "'original/*' 'metal/*'" and returns
// a slice of individual patterns without quotes: ["original/*", "metal/*"].
func parseExcludePatterns(exclude string) []string {
	if exclude == "" {
		return nil
	}
	var patterns []string
	current := ""
	inQuote := false

	for i := 0; i < len(exclude); i++ {
		ch := exclude[i]
		if ch == '\'' || ch == '"' {
			if inQuote {
				// End of quoted pattern
				if current != "" {
					patterns = append(patterns, current)
					current = ""
				}
				inQuote = false
			} else {
				// Start of quoted pattern
				inQuote = true
			}
		} else if inQuote {
			current += string(ch)
		}
		// Skip whitespace outside quotes
	}

	// Handle any remaining pattern
	if current != "" {
		patterns = append(patterns, current)
	}

	return patterns
}

// generateHFSingleFileDownloadScript downloads a single file from a Hugging Face
// repository deterministically. filePath is the relative path inside the repo.
func generateHFSingleFileDownloadScript(namespace, model, revision, filePath string) string {
	return fmt.Sprintf(`set -euo pipefail
if [ -f /run/secrets/hf-token ]; then export HF_TOKEN="$(cat /run/secrets/hf-token)"; fi
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
