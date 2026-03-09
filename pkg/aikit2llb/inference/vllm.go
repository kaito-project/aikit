package inference

import (
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
)

// installVLLMDependencies installs Python dependencies and a C compiler required for vLLM backend.
// vLLM's Triton kernels need a C compiler (gcc) for JIT compilation at runtime.
func installVLLMDependencies(s llb.State, merge llb.State) llb.State {
	merge = installPythonBaseDependencies(s, merge)

	savedState := s
	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev && apt-get clean"),
		llb.WithCustomName("Installing C compiler for vLLM Triton JIT"),
	).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
