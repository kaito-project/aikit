package packager

import (
	"fmt"
	"path"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/moby/buildkit/client/llb"
)

// resolveSourceState normalizes a model/artifact source reference into an llb.State.
// Supports local context ("." or "context"), HTTP(S), huggingface://, or a path/glob
// inside the local context. For HTTP(S) single files, preserveHTTPFilename controls
// whether the original basename is explicitly enforced (useful to avoid anonymous temp names).
// exclude is an optional space-separated list of patterns to exclude from huggingface downloads.
// HF token secret is automatically mounted if available in the BuildKit session.
func resolveSourceState(source, sessionID string, preserveHTTPFilename bool, exclude string) (llb.State, error) {
	if source == "" || source == "." || source == "context" {
		return llb.Local(localNameContext, llb.SessionID(sessionID), llb.SharedKeyHint(localNameContext)), nil
	}
	switch {
	case strings.HasPrefix(source, "https://") || strings.HasPrefix(source, "http://"):
		if preserveHTTPFilename {
			base := path.Base(source)
			return llb.HTTP(source, llb.Filename(base)), nil
		}
		return llb.HTTP(source), nil
	case strings.HasPrefix(source, "huggingface://"):
		// If the reference includes a file path (namespace/model/file...), fetch only that file.
		trimmed := strings.TrimPrefix(source, "huggingface://")
		if strings.Count(trimmed, "/") >= 2 { // namespace/model/file (optionally with further subdirs)
			if spec, err := inference.ParseHuggingFaceSpec(source); err == nil && spec.SubPath != "" {
				// Use hf CLI to download only the specified file (deterministic & token aware)
				fileScript := generateHFSingleFileDownloadScript(spec.Namespace, spec.Model, spec.Revision, spec.SubPath)
				runOpts := []llb.RunOption{
					llb.Args([]string{"bash", "-c", fileScript}),
					llb.AddSecret("/run/secrets/hf-token", llb.SecretID("hf-token"), llb.SecretOptional),
				}
				run := llb.Image(hfCLIImage).Run(runOpts...)
				return llb.Scratch().File(llb.Copy(run.Root(), "/out/", "/", &llb.CopyInfo{CopyDirContentsOnly: true})), nil
			}
		}
		// Fallback: download full repository snapshot
		st, err := buildHuggingFaceState(source, exclude)
		if err != nil {
			return llb.State{}, fmt.Errorf("failed to build huggingface state for %q: %w", source, err)
		}
		return st, nil
	default:
		include := source
		if strings.HasSuffix(include, "/") {
			include += "**"
		}
		return llb.Local(localNameContext,
			llb.IncludePatterns([]string{include}),
			llb.SessionID(sessionID),
			llb.SharedKeyHint(localNameContext+":"+include),
		), nil
	}
}

// debug helper (not currently used in production code) to validate error paths.
func mustResolve(source, sessionID string, preserve bool) llb.State { //nolint:unused
	st, err := resolveSourceState(source, sessionID, preserve, "")
	if err != nil {
		panic(fmt.Sprintf("resolve failed: %v", err))
	}
	return st
}
