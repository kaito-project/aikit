package inference

import (
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
)

// installVLLMDependencies installs dependencies required for vllm backend.
func installVLLMDependencies(s llb.State, merge llb.State) llb.State {
	savedState := s

	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y git python3 python3-pip python3-venv python-is-python3 && pip install uv && pip install grpcio-tools==1.71.0 --no-dependencies && pip install flash-attn==2.8.2 && apt-get clean"), llb.IgnoreCache).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
