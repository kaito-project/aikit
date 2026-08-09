package inference

import (
	"context"
	"strings"
	"testing"

	"github.com/moby/buildkit/client/llb"
)

const (
	parakeetAptCleanupFragment = "rm -rf /var/lib/apt/lists/*"
	parakeetPythonFragment     = "python3"
)

func TestInstallParakeetCppDependenciesAddsOnlyAudioConverter(t *testing.T) {
	baseState := llb.Image("ubuntu:22.04")
	result := installParakeetCppDependencies(baseState, baseState)

	definition, err := result.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal parakeet.cpp dependencies: %v", err)
	}

	ffmpegInstall := findInferenceExecOp(t, definition, "ffmpeg")
	command := strings.Join(ffmpegInstall.op.GetExec().Meta.Args, "\x00")
	for _, fragment := range []string{
		"apt-get install --no-install-recommends -y ffmpeg",
		parakeetAptCleanupFragment,
		"/var/cache/apt/archives/*",
	} {
		if !strings.Contains(command, fragment) {
			t.Fatalf("parakeet.cpp dependency command = %q, want %q", command, fragment)
		}
	}
	for _, fragment := range []string{
		parakeetPythonFragment,
		"python3-pip",
		"gcc",
		"libc6-dev",
		"cuda",
		"rocm",
	} {
		if strings.Contains(command, fragment) {
			t.Fatalf("parakeet.cpp dependency command = %q, unexpectedly contains %q", command, fragment)
		}
	}
}
