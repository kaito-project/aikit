package inference

import (
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
)

// installVLLMDependencies installs the host C compiler required by vLLM's
// Triton runtime compilation. The backend artifact already contains portable
// Python, its virtual environment, generated gRPC bindings, and CUDA runtime
// libraries, so installing a second Python or CUDA environment is unnecessary.
func installVLLMDependencies(s llb.State, merge llb.State) llb.State {
	savedState := s
	s = s.Run(
		utils.Sh("apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*"),
		llb.WithCustomName("Installing C compiler for vLLM Triton JIT"),
		llb.IgnoreCache,
	).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
