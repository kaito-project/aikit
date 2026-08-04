package inference

import (
	"context"
	"strings"
	"testing"

	"github.com/moby/buildkit/client/llb"
)

func TestInstallVLLMDependenciesOnlyAddsRuntimeCompiler(t *testing.T) {
	baseState := llb.Image("ubuntu:22.04")
	result := installVLLMDependencies(baseState, baseState)

	definition, err := result.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal vLLM dependencies: %v", err)
	}

	compilerInstall := findInferenceExecOp(t, definition, "gcc libc6-dev")
	command := strings.Join(compilerInstall.op.GetExec().Meta.Args, "\x00")
	for _, fragment := range []string{
		"gcc libc6-dev",
		"rm -rf /var/lib/apt/lists/*",
		"/var/cache/apt/archives/*",
	} {
		if !strings.Contains(command, fragment) {
			t.Fatalf("vLLM dependency command = %q, want %q", command, fragment)
		}
	}
	for _, fragment := range []string{
		"python3",
		"python3-pip",
		"python3-venv",
		"grpcio-tools",
		"pip install",
		"libcublas",
		"cuda-cudart",
		"pciutils",
	} {
		if strings.Contains(command, fragment) {
			t.Fatalf("vLLM dependency command = %q, unexpectedly contains %q", command, fragment)
		}
	}
}
