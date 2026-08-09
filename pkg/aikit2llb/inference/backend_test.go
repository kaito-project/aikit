package inference

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/solver/pb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	testCPULlamaCppBackend = "cpu-llama-cpp"
	testMinimumCUDA12      = "12.0"
	testLocalAIVersion     = "v4.8.2"
)

func TestResolveBackendCurrentCompatibility(t *testing.T) {
	tests := []struct {
		name            string
		config          *config.InferenceConfig
		platform        specs.Platform
		wantName        string
		wantVersion     string
		wantProfile     backendcatalog.TargetProfile
		wantRunner      backendcatalog.RunnerProfile
		wantFallbacks   int
		wantMinimumCUDA string
	}{
		{
			name:        "default CPU llama-cpp amd64",
			config:      &config.InferenceConfig{},
			platform:    specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:    testCPULlamaCppBackend,
			wantVersion: testLocalAIVersion,
			wantProfile: backendcatalog.TargetProfileCPU,
			wantRunner:  backendcatalog.RunnerProfileLlamaCpp,
		},
		{
			name:        "default CPU llama-cpp arm64",
			config:      &config.InferenceConfig{},
			platform:    specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64},
			wantName:    testCPULlamaCppBackend,
			wantVersion: testLocalAIVersion,
			wantProfile: backendcatalog.TargetProfileCPU,
			wantRunner:  backendcatalog.RunnerProfileLlamaCpp,
		},
		{
			name: "CUDA llama-cpp keeps explicit CPU fallback",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeNVIDIA,
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:        "cuda12-llama-cpp",
			wantVersion:     testLocalAIVersion,
			wantProfile:     backendcatalog.TargetProfileCUDA12,
			wantRunner:      backendcatalog.RunnerProfileLlamaCpp,
			wantFallbacks:   1,
			wantMinimumCUDA: testMinimumCUDA12,
		},
		{
			name: "CUDA diffusers uses promoted release entry",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendDiffusers},
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:        "cuda12-diffusers",
			wantVersion:     testLocalAIVersion,
			wantProfile:     backendcatalog.TargetProfileCUDA12,
			wantRunner:      backendcatalog.RunnerProfileHFConfig,
			wantMinimumCUDA: testMinimumCUDA12,
		},
		{
			name: "CUDA vllm",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendVLLM},
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:        "cuda12-vllm",
			wantVersion:     testLocalAIVersion,
			wantProfile:     backendcatalog.TargetProfileCUDA12,
			wantRunner:      backendcatalog.RunnerProfileHFConfig,
			wantMinimumCUDA: testMinimumCUDA12,
		},
		{
			name: "CPU vllm-cpp arm64",
			config: &config.InferenceConfig{
				Backends: []string{utils.BackendVLLMCpp},
			},
			platform:    specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64},
			wantName:    "cpu-vllm-cpp",
			wantVersion: testLocalAIVersion,
			wantProfile: backendcatalog.TargetProfileCPU,
			wantRunner:  backendcatalog.RunnerProfileVLLMCpp,
		},
		{
			name: "CUDA vllm-cpp selects upstream nvidia default",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendVLLMCpp},
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:        "cuda13-vllm-cpp",
			wantVersion:     testLocalAIVersion,
			wantProfile:     backendcatalog.TargetProfileCUDA13,
			wantRunner:      backendcatalog.RunnerProfileVLLMCpp,
			wantMinimumCUDA: "13.0",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resolved, err := ResolveBackend(test.config, test.platform)
			if err != nil {
				t.Fatalf("ResolveBackend() error = %v", err)
			}
			if resolved.Backend.InstallName != test.wantName {
				t.Errorf("install name = %q, want %q", resolved.Backend.InstallName, test.wantName)
			}
			if resolved.Version != test.wantVersion {
				t.Errorf("version = %q, want %q", resolved.Version, test.wantVersion)
			}
			if resolved.TargetProfile != test.wantProfile {
				t.Errorf("target profile = %q, want %q", resolved.TargetProfile, test.wantProfile)
			}
			if resolved.RunnerProfile != test.wantRunner {
				t.Errorf("runner profile = %q, want %q", resolved.RunnerProfile, test.wantRunner)
			}
			if len(resolved.Fallbacks) != test.wantFallbacks {
				t.Errorf("fallbacks = %d, want %d", len(resolved.Fallbacks), test.wantFallbacks)
			}
			if resolved.MinimumCUDA != test.wantMinimumCUDA {
				t.Errorf("minimum CUDA = %q, want %q", resolved.MinimumCUDA, test.wantMinimumCUDA)
			}
			for _, ref := range []string{resolved.Core.Ref, resolved.Backend.Ref} {
				if !strings.Contains(ref, "@sha256:") || strings.Contains(ref, ":latest") {
					t.Errorf("artifact ref is not immutable: %q", ref)
				}
			}
		})
	}
}

func TestResolveBackendFailsClosed(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	tests := []struct {
		name   string
		config *config.InferenceConfig
	}{
		{
			name:   "unknown family does not become llama-cpp",
			config: &config.InferenceConfig{Backends: []string{"unknown"}},
		},
		{
			name: "unsupported exact selector does not use nvidia default",
			config: &config.InferenceConfig{
				Runtime:           utils.RuntimeNVIDIA,
				Backends:          []string{utils.BackendVLLMCpp},
				BackendCapability: "nvidia-cuda-12",
			},
		},
		{
			name: "runtime mismatch is rejected",
			config: &config.InferenceConfig{
				Backends:          []string{utils.BackendLlamaCpp},
				BackendCapability: "nvidia",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ResolveBackend(test.config, platform)
			if err == nil {
				t.Fatal("ResolveBackend() succeeded, want error")
			}
			if !stderrors.Is(err, backendcatalog.ErrNotFound) && !strings.Contains(err.Error(), "requires runtime") {
				t.Fatalf("ResolveBackend() error = %v, want exact resolution failure", err)
			}
		})
	}
}

func TestInstallBackendsUsesOnlyCatalogArtifacts(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	cfg := &config.InferenceConfig{Runtime: utils.RuntimeNVIDIA}
	resolved, err := ResolveBackend(cfg, platform)
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}

	base := llb.Image(utils.UbuntuBase, llb.Platform(platform))
	state := installBackends(resolved, platform, base, base)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal backend definition: %v", err)
	}

	wantRefs := map[string]bool{resolved.Backend.Ref: false}
	for _, fallback := range resolved.Fallbacks {
		wantRefs[fallback.Ref] = false
	}
	metadataFiles := 0
	primaryMetadataFiles := 0
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}
		if source := op.GetSource(); source != nil {
			for ref := range wantRefs {
				if strings.Contains(source.Identifier, ref) {
					wantRefs[ref] = true
				}
			}
			if strings.Contains(source.Identifier, "local-ai-backends:") {
				t.Errorf("backend source reconstructed a mutable tag: %q", source.Identifier)
			}
		}
		file := op.GetFile()
		if file == nil {
			continue
		}
		for _, action := range file.Actions {
			mkfile := action.GetMkfile()
			if mkfile == nil || !strings.HasSuffix(mkfile.Path, "/metadata.json") {
				continue
			}
			var metadata backendMetadata
			if err := json.Unmarshal(mkfile.Data, &metadata); err != nil {
				t.Fatalf("unmarshal backend metadata: %v", err)
			}
			if metadata.CatalogDigest != resolved.CatalogDigest || metadata.Artifact == "" {
				t.Errorf("metadata = %+v, want catalog and artifact digests", metadata)
			}
			if metadata.Artifact == resolved.Backend.Ref {
				primaryMetadataFiles++
				if metadata.SourceRef != resolved.SourceRef || metadata.Version != resolved.Version || metadata.Selector != string(resolved.Selector) || metadata.Status != string(resolved.Status) {
					t.Errorf("primary metadata = %+v, want selected entry provenance", metadata)
				}
			} else if metadata.SourceRef != "" || metadata.Version != "" || metadata.Selector != "" || metadata.Status != "" {
				t.Errorf("fallback metadata contains primary-only provenance: %+v", metadata)
			}
			metadataFiles++
		}
	}

	for ref, found := range wantRefs {
		if !found {
			t.Errorf("catalog artifact source %q is missing", ref)
		}
	}
	if metadataFiles != 1+len(resolved.Fallbacks) {
		t.Errorf("metadata files = %d, want %d", metadataFiles, 1+len(resolved.Fallbacks))
	}
	if primaryMetadataFiles != 1 {
		t.Errorf("primary metadata files = %d, want 1", primaryMetadataFiles)
	}
}

func TestInstallVLLMBackendKeepsCopyAndMetadataInOneFileOp(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	cfg := &config.InferenceConfig{Runtime: utils.RuntimeNVIDIA, Backends: []string{utils.BackendVLLM}}
	resolved, err := ResolveBackend(cfg, platform)
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}

	base := llb.Image(utils.UbuntuBase, llb.Platform(platform))
	state := installBackends(resolved, platform, base, base)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal vLLM backend definition: %v", err)
	}

	backendDir := "/backends/" + resolved.Backend.InstallName
	fileOps := 0
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}
		file := op.GetFile()
		if file == nil {
			continue
		}
		var hasCopy, hasMetadata bool
		for _, action := range file.Actions {
			if copyAction := action.GetCopy(); copyAction != nil && copyAction.Src == "/" && strings.HasPrefix(copyAction.Dest, backendDir) {
				hasCopy = true
			}
			if mkfile := action.GetMkfile(); mkfile != nil && mkfile.Path == backendDir+"/metadata.json" {
				hasMetadata = true
			}
		}
		if hasCopy || hasMetadata {
			fileOps++
			if !hasCopy || !hasMetadata {
				t.Fatal("backend copy and metadata are not chained in one file operation")
			}
		}
	}
	if fileOps != 1 {
		t.Errorf("backend file operations = %d, want 1", fileOps)
	}
}
