package packager

import (
	"fmt"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/moby/buildkit/client/llb"
)

// buildHuggingFaceState returns an llb.State containing the downloaded Hugging Face
// repository snapshot rooted at /. It mounts an optional secret containing the token
// when hfSecretFlag is non-empty (flag indicates user requested secret mount).
func buildHuggingFaceState(source string, hfSecretFlag string) (llb.State, error) {
	if !strings.HasPrefix(source, "huggingface://") {
		return llb.State{}, fmt.Errorf("not a huggingface source: %s", source)
	}
	spec, err := inference.ParseHuggingFaceSpec(source)
	if err != nil {
		return llb.State{}, fmt.Errorf("invalid huggingface source: %w", err)
	}
	dlScript := generateHFDownloadScript(spec.Namespace, spec.Model, spec.Revision)
	runOpts := []llb.RunOption{llb.Args([]string{"bash", "-c", dlScript})}
	if hfSecretFlag != "" { // presence acts as opt-in to secret mount
		runOpts = append(runOpts, llb.AddSecret("/run/secrets/hf-token", llb.SecretID("hf-token")))
	}
	run := llb.Image(hfCLIImage).Run(runOpts...)
	return llb.Scratch().File(llb.Copy(run.Root(), "/out/", "/")), nil
}
