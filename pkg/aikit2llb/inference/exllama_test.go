package inference

import (
	"testing"

	"github.com/moby/buildkit/client/llb"
)

func TestInstallExllamaDependencies(t *testing.T) {
	// Create a simple base state for testing
	baseState := llb.Image("ubuntu:22.04")
	mergeState := baseState

	// Call the function to install dependencies
	// This should execute without panicking
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("installExllamaDependencies panicked: %v", r)
		}
	}()

	result := installExllamaDependencies(baseState, mergeState)

	// The function should return a valid LLB state
	// We can't easily test the actual installation without running BuildKit,
	// but we can verify the function executes without panicking
	_ = result // Use the result to avoid unused variable warning
}

func TestInstallPythonBaseDependencies(t *testing.T) {
	// Create a simple base state for testing
	baseState := llb.Image("ubuntu:22.04")
	mergeState := baseState

	// Call the function to install dependencies
	// This should execute without panicking
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("installPythonBaseDependencies panicked: %v", r)
		}
	}()

	result := installPythonBaseDependencies(baseState, mergeState)

	// The function should return a valid LLB state
	// We can't easily test the actual installation without running BuildKit,
	// but we can verify the function executes without panicking
	_ = result // Use the result to avoid unused variable warning
}
