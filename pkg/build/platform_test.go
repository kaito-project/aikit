package build

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/pb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const testBuildModelName = "test"

func TestBuildInferenceUsesBuildPlatformForArtifactHelpers(t *testing.T) {
	buildPlatform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	gatewayClient := &recordingBuildClient{
		buildOpts: client.BuildOpts{
			Opts: map[string]string{
				keyTargetPlatform: "linux/amd64,linux/arm64",
			},
			Workers: []client.WorkerInfo{{Platforms: []specs.Platform{buildPlatform}}},
			LLBCaps: pb.Caps.CapSet(pb.Caps.All()),
		},
	}
	cfg := &config.InferenceConfig{
		APIVersion: utils.APIv1alpha1,
		Models: []config.Model{{
			Name:   testBuildModelName,
			Source: "oci://example.com/models/test:latest",
		}},
	}

	result, err := buildInference(context.Background(), gatewayClient, cfg)
	if err != nil {
		t.Fatalf("build inference: %v", err)
	}

	definitions := gatewayClient.solveDefinitions()
	if len(definitions) != 2 {
		t.Fatalf("solve definitions = %d, want 2", len(definitions))
	}

	seenTargets := make(map[string]bool, len(definitions))
	for _, definition := range definitions {
		helperSource := findBuildSourceOp(t, definition, "ghcr.io/oras-project/oras")
		assertBuildOpPlatform(t, helperSource, buildPlatform)

		modelPull := findBuildExecOp(t, definition, "example.com/models/test:latest")
		assertBuildOpPlatform(t, modelPull, buildPlatform)

		localAIPull := findBuildExecOp(t, definition, "ghcr.io/kaito-project/aikit/localai:")
		assertBuildOpPlatform(t, localAIPull, buildPlatform)

		targetArchitecture := artifactArchitecture(t, localAIPull)
		targetPlatform := specs.Platform{OS: utils.PlatformLinux, Architecture: targetArchitecture}
		seenTargets[targetArchitecture] = true

		backendSource := findBuildSourceOp(t, definition, utils.BackendOCIRegistry)
		assertBuildOpPlatform(t, backendSource, targetPlatform)

		baseSource := findBuildSourceOp(t, definition, "ubuntu:22.04")
		assertBuildOpPlatform(t, baseSource, targetPlatform)
	}

	for _, architecture := range []string{utils.PlatformAMD64, utils.PlatformARM64} {
		if !seenTargets[architecture] {
			t.Errorf("target architecture %s was not built", architecture)
		}
	}

	if result.Ref != nil {
		t.Fatal("multi-platform result unexpectedly has a single reference")
	}
	if len(result.Refs) != 2 {
		t.Fatalf("multi-platform references = %d, want 2", len(result.Refs))
	}

	var exportPlatforms exptypes.Platforms
	if err := json.Unmarshal(result.Metadata[exptypes.ExporterPlatformsKey], &exportPlatforms); err != nil {
		t.Fatalf("unmarshal exporter platforms: %v", err)
	}
	if len(exportPlatforms.Platforms) != 2 {
		t.Fatalf("export platforms = %d, want 2", len(exportPlatforms.Platforms))
	}

	wantPlatforms := []specs.Platform{
		{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
		{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64},
	}
	for i, wantPlatform := range wantPlatforms {
		gotPlatform := exportPlatforms.Platforms[i]
		wantID := fmt.Sprintf("%s/%s", wantPlatform.OS, wantPlatform.Architecture)
		if gotPlatform.ID != wantID || !reflect.DeepEqual(gotPlatform.Platform, wantPlatform) {
			t.Errorf("export platform %d = %#v, want ID %q and platform %#v", i, gotPlatform, wantID, wantPlatform)
		}
		if _, ok := result.Refs[wantID]; !ok {
			t.Errorf("reference for %s is missing", wantID)
		}

		configKey := fmt.Sprintf("%s/%s", exptypes.ExporterImageConfigKey, wantID)
		var image specs.Image
		if err := json.Unmarshal(result.Metadata[configKey], &image); err != nil {
			t.Fatalf("unmarshal image config for %s: %v", wantID, err)
		}
		if !reflect.DeepEqual(image.Platform, wantPlatform) {
			t.Errorf("image platform for %s = %#v, want %#v", wantID, image.Platform, wantPlatform)
		}
	}
}

type recordingBuildClient struct {
	client.Client

	buildOpts     client.BuildOpts
	mu            sync.Mutex
	definitions   []*pb.Definition
	nextReference int
}

func (c *recordingBuildClient) BuildOpts() client.BuildOpts {
	return c.buildOpts
}

func (c *recordingBuildClient) Solve(_ context.Context, request client.SolveRequest) (*client.Result, error) {
	c.mu.Lock()
	c.definitions = append(c.definitions, request.Definition.CloneVT())
	c.nextReference++
	reference := &recordingBuildReference{id: c.nextReference}
	c.mu.Unlock()

	result := client.NewResult()
	result.SetRef(reference)
	return result, nil
}

func (c *recordingBuildClient) solveDefinitions() []*pb.Definition {
	c.mu.Lock()
	defer c.mu.Unlock()

	definitions := make([]*pb.Definition, len(c.definitions))
	for i, definition := range c.definitions {
		definitions[i] = definition.CloneVT()
	}
	return definitions
}

type recordingBuildReference struct {
	client.Reference
	id int
}

func findBuildSourceOp(t *testing.T, definition *pb.Definition, identifierFragment string) *pb.Op {
	t.Helper()

	var matches []*pb.Op
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}
		if source := op.GetSource(); source != nil && strings.Contains(source.Identifier, identifierFragment) {
			matches = append(matches, op)
		}
	}
	if len(matches) != 1 {
		t.Fatalf("source ops containing %q = %d, want 1", identifierFragment, len(matches))
	}
	return matches[0]
}

func findBuildExecOp(t *testing.T, definition *pb.Definition, commandFragment string) *pb.Op {
	t.Helper()

	var matches []*pb.Op
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}
		if exec := op.GetExec(); exec != nil && strings.Contains(strings.Join(exec.Meta.Args, "\x00"), commandFragment) {
			matches = append(matches, op)
		}
	}
	if len(matches) != 1 {
		t.Fatalf("exec ops containing %q = %d, want 1", commandFragment, len(matches))
	}
	return matches[0]
}

func assertBuildOpPlatform(t *testing.T, op *pb.Op, want specs.Platform) {
	t.Helper()

	if op.Platform == nil {
		t.Fatalf("operation platform is nil, want %#v", want)
	}
	got := specs.Platform{
		OS:           op.Platform.OS,
		Architecture: op.Platform.Architecture,
		Variant:      op.Platform.Variant,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("operation platform = %#v, want %#v", got, want)
	}
}

func artifactArchitecture(t *testing.T, localAIPull *pb.Op) string {
	t.Helper()

	command := strings.Join(localAIPull.GetExec().Meta.Args, "\x00")
	for _, architecture := range []string{utils.PlatformAMD64, utils.PlatformARM64} {
		if strings.Contains(command, "-"+architecture) {
			return architecture
		}
	}
	t.Fatalf("LocalAI artifact command does not select a supported target architecture: %q", command)
	return ""
}
