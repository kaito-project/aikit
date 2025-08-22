package inference

import (
	"github.com/moby/buildkit/client/llb"
)

// installDiffusersDependencies installs minimal Python dependencies required for diffusers backend.
// Diffusers only needs basic Python tools, no build dependencies.
func installDiffusersDependencies(s llb.State, merge llb.State) llb.State {
	return installPythonBaseDependencies(s, merge)
}
