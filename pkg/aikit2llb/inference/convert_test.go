package inference

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	testInferenceModelName   = "test"
	testInferenceModelSource = "model.gguf"
)

func TestAikit2LLBWithPlatformsSeparatesHelperAndTargetPlatforms(t *testing.T) {
	buildPlatform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	targetPlatform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	cfg := &config.InferenceConfig{
		Runtime:  utils.RuntimeNVIDIA,
		Backends: []string{utils.BackendLlamaCpp},
		Models: []config.Model{{
			Name:   testInferenceModelName,
			Source: "oci://example.com/models/test:latest",
		}},
	}

	state, image, err := Aikit2LLBWithPlatforms(cfg, buildPlatform, targetPlatform)
	if err != nil {
		t.Fatalf("convert inference config: %v", err)
	}
	if !reflect.DeepEqual(image.Platform, *targetPlatform) {
		t.Fatalf("image platform = %#v, want %#v", image.Platform, *targetPlatform)
	}

	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal inference definition: %v", err)
	}

	helperSource := findInferenceSourceOp(t, definition, orasImage)
	assertInferenceOpPlatform(t, helperSource, *buildPlatform)

	modelPull := findInferenceExecOp(t, definition, "example.com/models/test:latest")
	assertInferenceOpPlatform(t, modelPull, *buildPlatform)
	modelPullCommand := strings.Join(modelPull.op.GetExec().Meta.Args, "\x00")
	for _, fragment := range []string{
		`oras resolve  --full-reference "$ref"`,
		`oras manifest fetch  "$pinned_ref"`,
		`.platform.os != "unknown"`,
		`.platform.architecture != "unknown"`,
		`unique | length) > 1`,
		`vnd.docker.reference.type`,
		`attestation-manifest`,
		`platform_flag="--platform linux/amd64"`,
		`oras pull  $platform_flag "$pinned_ref"`,
	} {
		if !strings.Contains(modelPullCommand, fragment) {
			t.Fatalf("OCI model pull command = %q, want %q", modelPullCommand, fragment)
		}
	}

	backend, err := ResolveBackend(cfg, *targetPlatform)
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}
	localAIPull := findInferenceExecOp(t, definition, backend.Core.Ref)
	assertInferenceOpPlatform(t, localAIPull, *buildPlatform)
	if command := strings.Join(localAIPull.op.GetExec().Meta.Args, "\x00"); !strings.Contains(command, "@sha256:") {
		t.Fatalf("LocalAI artifact command = %q, want digest-qualified artifact", command)
	}

	runtimeBaseSource := findInferenceSourceOp(t, definition, backend.RuntimeBase.Ref)
	assertInferenceOpPlatform(t, runtimeBaseSource, *targetPlatform)

	backendSources := findInferenceSourceOps(t, definition, utils.BackendOCIRegistry)
	if len(backendSources) == 0 {
		t.Fatal("backend target image source is missing")
	}
	for _, backendSource := range backendSources {
		assertInferenceOpPlatform(t, backendSource, *targetPlatform)
	}
}

func TestAikit2LLBUsesCatalogRuntimeBaseForArbitraryFamily(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	backend := testArbitraryBackendPlan(*platform)
	backend.SystemPackages = nil
	cfg := &config.InferenceConfig{
		Backends: []string{backend.Family},
		Models: []config.Model{{
			Name: testInferenceModelName, Source: testInferenceModelSource,
		}},
	}

	state, _, err := aikit2LLBWithResolvedBackend(cfg, platform, platform, backend)
	if err != nil {
		t.Fatalf("convert inference config: %v", err)
	}
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal inference definition: %v", err)
	}

	runtimeBase := findInferenceSourceOp(t, definition, backend.RuntimeBase.Ref)
	assertInferenceOpPlatform(t, runtimeBase, *platform)
}

func TestAikit2LLBUsesRunnerRuntimeBaseOnlyForRunnerMode(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	backend := testArbitraryBackendPlan(*platform)
	backend.RunnerProfile = backendcatalog.RunnerProfileHFConfig
	runnerRuntimeBase := backendcatalog.Artifact{
		Ref: "registry.example/runner-base@sha256:6666666666666666666666666666666666666666666666666666666666666666",
	}
	backend.RunnerRuntimeBase = &runnerRuntimeBase
	backend.SystemPackages = nil

	tests := []struct {
		name       string
		config     *config.InferenceConfig
		wantBase   string
		absentBase string
	}{
		{
			name: "standard image",
			config: &config.InferenceConfig{
				Backends: []string{backend.Family},
				Models:   []config.Model{{Name: testInferenceModelName, Source: testInferenceModelSource}},
			},
			wantBase:   backend.RuntimeBase.Ref,
			absentBase: runnerRuntimeBase.Ref,
		},
		{
			name:       "runner image",
			config:     &config.InferenceConfig{Backends: []string{backend.Family}},
			wantBase:   runnerRuntimeBase.Ref,
			absentBase: backend.RuntimeBase.Ref,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state, image, err := aikit2LLBWithResolvedBackend(test.config, platform, platform, backend)
			if err != nil {
				t.Fatalf("convert inference config: %v", err)
			}
			definition, err := state.Marshal(context.Background())
			if err != nil {
				t.Fatalf("marshal inference definition: %v", err)
			}
			assertInferenceOpPlatform(t, findInferenceSourceOp(t, definition, test.wantBase), *platform)
			if matches := findInferenceSourceOps(t, definition, test.absentBase); len(matches) != 0 {
				t.Fatalf("unexpected runtime base %q appears %d times", test.absentBase, len(matches))
			}
			if got := image.Config.Labels["ai.kaito.aikit.runtime-base.artifact"]; got != test.wantBase {
				t.Errorf("runtime base label = %q, want %q", got, test.wantBase)
			}
		})
	}
}

func TestAikit2LLBPreservesSinglePlatformBehavior(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	cfg := &config.InferenceConfig{
		Backends: []string{utils.BackendLlamaCpp},
		Models: []config.Model{{
			Name:   testInferenceModelName,
			Source: "oci://example.com/models/test:latest",
		}},
	}

	legacyState, legacyImage, err := Aikit2LLB(cfg, platform)
	if err != nil {
		t.Fatalf("convert with compatible API: %v", err)
	}
	explicitState, explicitImage, err := Aikit2LLBWithPlatforms(cfg, platform, platform)
	if err != nil {
		t.Fatalf("convert with explicit platforms: %v", err)
	}

	legacyDefinition, err := legacyState.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal compatible definition: %v", err)
	}
	explicitDefinition, err := explicitState.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal explicit definition: %v", err)
	}
	legacyHead, err := legacyDefinition.Head()
	if err != nil {
		t.Fatalf("resolve compatible definition head: %v", err)
	}
	explicitHead, err := explicitDefinition.Head()
	if err != nil {
		t.Fatalf("resolve explicit definition head: %v", err)
	}
	if legacyHead != explicitHead {
		t.Errorf("compatible definition head = %s, want %s", legacyHead, explicitHead)
	}
	if !reflect.DeepEqual(legacyImage, explicitImage) {
		t.Errorf("compatible image config = %#v, want %#v", legacyImage, explicitImage)
	}
}

func TestAikit2LLBWithBackendRejectsForgedResolution(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	cfg := &config.InferenceConfig{
		Models: []config.Model{{Name: testInferenceModelName, Source: testInferenceModelSource}},
	}
	backend, err := ResolveBackend(cfg, *platform)
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}
	backend.Backend.Ref = strings.Split(backend.Backend.Ref, "@")[0] + "@sha256:" + strings.Repeat("f", 64)

	if _, _, err := Aikit2LLBWithBackend(cfg, platform, platform, backend); err == nil || !strings.Contains(err.Error(), "does not match the embedded catalog") {
		t.Fatalf("Aikit2LLBWithBackend() error = %v, want catalog mismatch", err)
	}
}

func findInferenceSourceOp(t *testing.T, definition *llb.Definition, identifierFragment string) inferenceDefinitionOp {
	t.Helper()

	matches := findInferenceSourceOps(t, definition, identifierFragment)
	if len(matches) != 1 {
		t.Fatalf("source ops containing %q = %d, want 1", identifierFragment, len(matches))
	}
	return matches[0]
}

func findInferenceSourceOps(t *testing.T, definition *llb.Definition, identifierFragment string) []inferenceDefinitionOp {
	t.Helper()

	var matches []inferenceDefinitionOp
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		if source := graphOp.op.GetSource(); source != nil && strings.Contains(source.Identifier, identifierFragment) {
			matches = append(matches, graphOp)
		}
	}
	return matches
}

func assertInferenceOpPlatform(t *testing.T, graphOp inferenceDefinitionOp, want specs.Platform) {
	t.Helper()

	if graphOp.op.Platform == nil {
		t.Fatalf("operation platform is nil, want %#v", want)
	}
	got := specs.Platform{
		OS:           graphOp.op.Platform.OS,
		Architecture: graphOp.op.Platform.Architecture,
		Variant:      graphOp.op.Platform.Variant,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("operation platform = %#v, want %#v", got, want)
	}
}
