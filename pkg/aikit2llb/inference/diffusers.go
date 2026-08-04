package inference

import "github.com/moby/buildkit/client/llb"

// installDiffusersDependencies returns the existing runtime unchanged. LocalAI
// Diffusers backend artifacts contain portable Python, a complete virtual
// environment, generated gRPC bindings, and their CUDA runtime libraries.
func installDiffusersDependencies(_ llb.State, merge llb.State) llb.State {
	return merge
}
