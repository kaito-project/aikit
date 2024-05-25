package inference

import (
	"github.com/moby/buildkit/client/llb"
	"github.com/sozercan/aikit/pkg/aikit/config"
	"github.com/sozercan/aikit/pkg/utils"
)

func installExllama(c *config.InferenceConfig, s llb.State, merge llb.State) llb.State {
	backend := utils.BackendExllama
	for b := range c.Backends {
		if c.Backends[b] == utils.BackendExllamaV2 {
			backend = utils.BackendExllamaV2
		}
	}

	savedState := s
	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y bash git ca-certificates python3-pip python3-dev python3-venv python-is-python3 make g++ curl && pip install uv grpcio-tools --break-system-packages && apt-get clean"), llb.IgnoreCache).Root()

	s = cloneLocalAI(s)

	s = s.Run(utils.Bashf("export BUILD_TYPE=cublas && cd /tmp/localai/backend/python/%[1]s && make %[1]s", backend)).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
