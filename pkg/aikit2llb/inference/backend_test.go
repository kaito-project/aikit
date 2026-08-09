package inference

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"slices"
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
	testLocalAIVersion     = "v4.8.2"
	testLegacyLocalAI      = "v3.12.1"
	testArbitraryFamily    = "arbitrary-family"
)

var (
	testCUDA12Environment = []string{
		"BUILD_TYPE=cublas",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_REQUIRE_CUDA=cuda>=12.0",
		"NVIDIA_VISIBLE_DEVICES=all",
	}
	testCUDA13Environment = []string{
		"BUILD_TYPE=cublas",
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_REQUIRE_CUDA=cuda>=13.0",
		"NVIDIA_VISIBLE_DEVICES=all",
	}
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
		wantPackages    []string
		wantEnvironment []string
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
			wantEnvironment: testCUDA12Environment,
		},
		{
			name: "CUDA diffusers preserves legacy default",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{utils.BackendDiffusers},
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:        "cuda12-diffusers",
			wantVersion:     testLegacyLocalAI,
			wantProfile:     backendcatalog.TargetProfileCUDA12,
			wantRunner:      backendcatalog.RunnerProfileHFConfig,
			wantEnvironment: testCUDA12Environment,
		},
		{
			name: "CUDA diffusers exposes promoted release through exact selector",
			config: &config.InferenceConfig{
				Runtime:           utils.RuntimeNVIDIA,
				Backends:          []string{utils.BackendDiffusers},
				BackendCapability: string(backendcatalog.SelectorNVIDIACUDA12),
				Models:            []config.Model{{Name: "test", Source: "test"}},
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64},
			wantName:        "cuda12-diffusers",
			wantVersion:     testLocalAIVersion,
			wantProfile:     backendcatalog.TargetProfileCUDA12,
			wantRunner:      backendcatalog.RunnerProfileUnsupported,
			wantEnvironment: testCUDA12Environment,
		},
		{
			name: "Apple Silicon llama-cpp preserves legacy default",
			config: &config.InferenceConfig{
				Runtime: utils.RuntimeAppleSilicon,
			},
			platform:        specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64},
			wantName:        "gpu-vulkan-llama-cpp",
			wantVersion:     testLegacyLocalAI,
			wantProfile:     backendcatalog.TargetProfileVulkan,
			wantRunner:      backendcatalog.RunnerProfileUnsupported,
			wantEnvironment: []string{"VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json"},
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
			wantPackages:    []string{"gcc", "libc6-dev"},
			wantEnvironment: testCUDA12Environment,
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
			wantEnvironment: testCUDA13Environment,
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
			if !slices.Equal(resolved.SystemPackages, test.wantPackages) {
				t.Errorf("system packages = %q, want %q", resolved.SystemPackages, test.wantPackages)
			}
			if !slices.Equal(resolved.Environment, test.wantEnvironment) {
				t.Errorf("environment = %q, want %q", resolved.Environment, test.wantEnvironment)
			}
			for _, ref := range []string{resolved.RuntimeBase.Ref, resolved.Core.Ref, resolved.Backend.Ref} {
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
	resolved := testArbitraryBackendPlan(platform)

	base := llb.Image(resolved.RuntimeBase.Ref, llb.Platform(platform))
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
			if metadata.Alias != resolved.Family || metadata.Name == "" || metadata.GalleryURL != pinnedGalleryURL(resolved.Source) || metadata.Version != resolved.Version {
				t.Errorf("metadata = %+v, want LocalAI compatibility fields", metadata)
			}
			if metadata.CatalogDigest != resolved.CatalogDigest || metadata.Artifact == "" || metadata.Digest == "" {
				t.Errorf("metadata = %+v, want catalog and artifact digests", metadata)
			}
			if metadata.Artifact == resolved.Backend.Ref {
				primaryMetadataFiles++
				if metadata.URI != resolved.SourceRef || metadata.SourceRef != resolved.SourceRef || metadata.Selector != string(resolved.Selector) || metadata.Status != string(resolved.Status) {
					t.Errorf("primary metadata = %+v, want selected entry provenance", metadata)
				}
			} else {
				if metadata.URI != metadata.Artifact {
					t.Errorf("fallback metadata URI = %q, want installed artifact %q", metadata.URI, metadata.Artifact)
				}
				if metadata.SourceRef != "" || metadata.Selector != "" || metadata.Status != "" {
					t.Errorf("fallback metadata contains primary-only provenance: %+v", metadata)
				}
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

func TestBackendMetadataMatchesLocalAIV482(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)
	tests := []struct {
		name       string
		artifact   backendcatalog.BackendArtifact
		primary    bool
		wantURI    string
		wantDigest string
	}{
		{
			name:       "primary",
			artifact:   resolved.Backend,
			primary:    true,
			wantURI:    resolved.SourceRef,
			wantDigest: "sha256:3333333333333333333333333333333333333333333333333333333333333333",
		},
		{
			name:       "fallback",
			artifact:   resolved.Fallbacks[0],
			wantURI:    resolved.Fallbacks[0].Ref,
			wantDigest: "sha256:4444444444444444444444444444444444444444444444444444444444444444",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metadataJSON := marshalBackendMetadata(resolved, test.artifact, test.primary)
			for iteration := 0; iteration < 10; iteration++ {
				if got := string(marshalBackendMetadata(resolved, test.artifact, test.primary)); got != string(metadataJSON) {
					t.Fatalf("metadata changed on serialization %d: got %q, want %q", iteration, got, metadataJSON)
				}
			}
			if !strings.HasSuffix(string(metadataJSON), "\n") {
				t.Fatal("metadata does not end with a newline")
			}

			var metadata map[string]string
			if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
				t.Fatalf("unmarshal backend metadata: %v", err)
			}
			want := map[string]string{
				"alias":          resolved.Family,
				"name":           test.artifact.InstallName,
				"gallery_url":    "github:example/catalog/backend/index.yaml@test-revision",
				"version":        resolved.Version,
				"uri":            test.wantURI,
				"digest":         test.wantDigest,
				"gallery_commit": resolved.Source.Revision,
				"catalog_digest": resolved.CatalogDigest,
				"artifact":       test.artifact.Ref,
			}
			for key, value := range want {
				got, ok := metadata[key]
				if !ok {
					t.Errorf("metadata is missing LocalAI field %q", key)
					continue
				}
				if got != value {
					t.Errorf("metadata %q = %q, want %q", key, got, value)
				}
			}
			if _, ok := metadata["installed_at"]; ok {
				t.Error("metadata unexpectedly contains nondeterministic installed_at")
			}
		})
	}
}

func TestInstallBackendKeepsCopyAndMetadataInOneFileOp(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)
	resolved.Fallbacks = nil
	resolved.SystemPackages = nil

	base := llb.Image(resolved.RuntimeBase.Ref, llb.Platform(platform))
	state := installBackends(resolved, platform, base, base)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal backend definition: %v", err)
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

func TestInstallBackendsUsesCatalogSystemPackagesForArbitraryFamily(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)
	base := llb.Image(resolved.RuntimeBase.Ref, llb.Platform(platform))

	state := installBackends(resolved, platform, base, base)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal backend definition: %v", err)
	}

	install := findInferenceExecOp(t, definition, "apt-get install --no-install-recommends -y gcc libc6-dev")
	command := strings.Join(install.op.GetExec().Meta.Args, "\x00")
	for _, fragment := range []string{
		"apt-get update",
		"apt-get install --no-install-recommends -y gcc libc6-dev",
		"apt-get clean",
		"rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*",
	} {
		if !strings.Contains(command, fragment) {
			t.Errorf("system package command = %q, want %q", command, fragment)
		}
	}
}

func TestInstallBackendsUsesCatalogRuntimeSymlinksForArbitraryFamily(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)
	base := llb.Image(resolved.RuntimeBase.Ref, llb.Platform(platform))

	state := installBackends(resolved, platform, base, base)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal backend definition: %v", err)
	}

	var matches int
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		file := graphOp.op.GetFile()
		if file == nil {
			continue
		}
		for _, action := range file.Actions {
			symlink := action.GetSymlink()
			if symlink == nil {
				continue
			}
			if symlink.Oldpath != resolved.RuntimeSymlinks[0].Target || symlink.Newpath != resolved.RuntimeSymlinks[0].Path {
				t.Fatalf("runtime symlink = %q -> %q, want %q -> %q", symlink.Newpath, symlink.Oldpath, resolved.RuntimeSymlinks[0].Path, resolved.RuntimeSymlinks[0].Target)
			}
			matches++
		}
	}
	if matches != 1 {
		t.Fatalf("runtime symlink actions = %d, want 1", matches)
	}
}

func testArbitraryBackendPlan(platform specs.Platform) backendcatalog.Resolution {
	const (
		runtimeBaseRef = "registry.example.com/runtime/base@sha256:1111111111111111111111111111111111111111111111111111111111111111"
		coreRef        = "registry.example.com/local-ai/core@sha256:2222222222222222222222222222222222222222222222222222222222222222"
		backendRef     = "registry.example.com/backends/arbitrary@sha256:3333333333333333333333333333333333333333333333333333333333333333"
		fallbackRef    = "registry.example.com/backends/fallback@sha256:4444444444444444444444444444444444444444444444444444444444444444"
	)

	return backendcatalog.Resolution{
		Entry: backendcatalog.Entry{
			Family:        testArbitraryFamily,
			Selector:      backendcatalog.SelectorDefault,
			Platform:      backendcatalog.Platform{OS: platform.OS, Architecture: platform.Architecture, Variant: platform.Variant},
			Runtime:       backendcatalog.RuntimeCPU,
			TargetProfile: backendcatalog.TargetProfileCPU,
			Status:        backendcatalog.StatusExperimental,
			Channel:       backendcatalog.ChannelStable,
			Version:       "v-test",
			SourceRef:     "registry.example.com/upstream/arbitrary:stable",
			RuntimeBase:   backendcatalog.Artifact{Ref: runtimeBaseRef},
			Core:          backendcatalog.Artifact{Ref: coreRef},
			Backend:       backendcatalog.BackendArtifact{Ref: backendRef, InstallName: "cpu-arbitrary-family"},
			Fallbacks: []backendcatalog.BackendArtifact{
				{Ref: fallbackRef, InstallName: "cpu-arbitrary-fallback"},
			},
			SystemPackages:  []string{"gcc", "libc6-dev"},
			RuntimeSymlinks: []backendcatalog.RuntimeSymlink{{Target: "libcompat.so.1", Path: "/usr/lib/libcompat.so.0"}},
			Environment:     []string{"ARBITRARY_ACCELERATOR=enabled", "ARBITRARY_CACHE=/var/cache/arbitrary"},
			RunnerProfile:   backendcatalog.RunnerProfileUnsupported,
		},
		CatalogDigest: "sha256:5555555555555555555555555555555555555555555555555555555555555555",
		Source: backendcatalog.Source{
			Repository: "https://github.com/example/catalog",
			Revision:   "test-revision",
			Path:       "backend/index.yaml",
		},
	}
}
