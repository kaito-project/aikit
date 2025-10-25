package inference

import (
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
)

// installPythonBaseDependencies installs minimal Python dependencies common to all Python backends.
func installPythonBaseDependencies(s llb.State, merge llb.State) llb.State {
	savedState := s

	// Install minimal Python dependencies common to all Python backends
	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y git python3 python3-pip python3-venv python-is-python3 && pip install uv && pip install grpcio-tools==1.71.0 --no-dependencies && apt-get clean"), llb.IgnoreCache).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

// installExllamaDependencies installs Python and other dependencies required for exllama2 backend.
// ExLLama2 needs additional build tools for compilation.
func installExllamaDependencies(s llb.State, merge llb.State) llb.State {
	savedState := s

	// Install Python and build dependencies needed for exllama2
	s = s.Run(utils.Sh("apt-get update && apt-get install --no-install-recommends -y bash git ca-certificates python3-pip python3-dev python3-venv python-is-python3 make g++ curl && pip install uv ninja && pip install grpcio-tools==1.71.0 --no-dependencies && apt-get clean"), llb.IgnoreCache).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
