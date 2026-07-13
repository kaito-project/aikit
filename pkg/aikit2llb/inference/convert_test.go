package inference

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const testInferenceModelName = "test"

func TestInstallRocmInstallsPciutilsForLlamaCpp(t *testing.T) {
	tests := []struct {
		name     string
		backends []string
	}{
		{
			name:     "implicit default llama-cpp backend",
			backends: nil,
		},
		{
			name:     "explicit llama-cpp backend",
			backends: []string{utils.BackendLlamaCpp},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.InferenceConfig{
				Runtime:  utils.RuntimeROCm,
				Backends: tt.backends,
			}

			base := llb.Image(utils.Ubuntu24Base)
			_, merged := installRocm(cfg, base, base)

			def, err := merged.Marshal(context.Background())
			if err != nil {
				t.Fatalf("marshal failed: %v", err)
			}

			combined := marshalDefinitionToString(def)
			wantInstall := "apt-get install -y pciutils rocm && apt-get clean"
			if !strings.Contains(combined, wantInstall) {
				t.Fatalf("expected ROCm install to contain %q, got: %s", wantInstall, combined)
			}
		})
	}
}

func marshalDefinitionToString(def *llb.Definition) string {
	if def == nil {
		return ""
	}

	var combined strings.Builder
	for _, d := range def.ToPB().Def {
		combined.Write(d)
	}

	return combined.String()
}

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
		`vnd.docker.reference.type`,
		`attestation-manifest`,
		`platform_flag="--platform linux/amd64"`,
		`oras pull  $platform_flag "$pinned_ref"`,
	} {
		if !strings.Contains(modelPullCommand, fragment) {
			t.Fatalf("OCI model pull command = %q, want %q", modelPullCommand, fragment)
		}
	}

	localAIPull := findInferenceExecOp(t, definition, localAIRepo)
	assertInferenceOpPlatform(t, localAIPull, *buildPlatform)
	if command := strings.Join(localAIPull.op.GetExec().Meta.Args, "\x00"); !strings.Contains(command, "-amd64") {
		t.Fatalf("LocalAI artifact command = %q, want amd64 artifact", command)
	}

	baseSource := findInferenceSourceOp(t, definition, "ubuntu:22.04")
	assertInferenceOpPlatform(t, baseSource, *targetPlatform)

	backendSources := findInferenceSourceOps(t, definition, utils.BackendOCIRegistry)
	if len(backendSources) == 0 {
		t.Fatal("backend target image source is missing")
	}
	for _, backendSource := range backendSources {
		assertInferenceOpPlatform(t, backendSource, *targetPlatform)
	}

	packageInstall := findInferenceExecOp(t, definition, "dpkg -i cuda-keyring_1.1-1_all.deb")
	assertInferenceOpPlatform(t, packageInstall, *targetPlatform)
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
