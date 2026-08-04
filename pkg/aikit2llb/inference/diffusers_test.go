package inference

import (
	"context"
	"testing"

	"github.com/moby/buildkit/client/llb"
)

func TestInstallDiffusersDependenciesDoesNotAddRuntimePackages(t *testing.T) {
	baseState := llb.Image("ubuntu:22.04")
	result := installDiffusersDependencies(baseState, baseState)

	baseDefinition, err := baseState.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal base state: %v", err)
	}
	resultDefinition, err := result.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal Diffusers dependencies: %v", err)
	}

	baseHead, err := baseDefinition.Head()
	if err != nil {
		t.Fatalf("resolve base head: %v", err)
	}
	resultHead, err := resultDefinition.Head()
	if err != nil {
		t.Fatalf("resolve Diffusers head: %v", err)
	}
	if resultHead != baseHead {
		t.Fatalf("Diffusers dependencies changed state head: got %s, want %s", resultHead, baseHead)
	}
}
