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
// hfSecretFlag indicates optional HF token secret mount for huggingface sources.
func resolveSourceState(source, sessionID, hfSecretFlag string, preserveHTTPFilename bool) (llb.State, error) {
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
				runOpts := []llb.RunOption{llb.Args([]string{"bash", "-c", fileScript})}
				if hfSecretFlag != "" {
					runOpts = append(runOpts, llb.AddSecret("/run/secrets/hf-token", llb.SecretID("hf-token")))
				}
				run := llb.Image(hfCLIImage).Run(runOpts...)
				return llb.Scratch().File(llb.Copy(run.Root(), "/out/", "/")), nil
			}
		}
		// Fallback: download full repository snapshot
		st, err := buildHuggingFaceState(source, hfSecretFlag)
		if err != nil {
			return llb.State{}, err
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
func mustResolve(source, sessionID, hfSecretFlag string, preserve bool) llb.State { //nolint:unused
	st, err := resolveSourceState(source, sessionID, hfSecretFlag, preserve)
	if err != nil {
		panic(fmt.Sprintf("resolve failed: %v", err))
	}
	return st
}
