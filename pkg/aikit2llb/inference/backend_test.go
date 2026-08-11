package inference

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"os"
	"os/exec"
	"path/filepath"
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
	testVLLMNativeSampler  = "VLLM_USE_FLASHINFER_SAMPLER=0"
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
			name: "exact CUDA 12 diffusers selects promoted release",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeCUDA12,
				Backends: []string{utils.BackendDiffusers},
				Models:   []config.Model{{Name: "test", Source: "test"}},
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
			wantEnvironment: slices.Concat(testCUDA12Environment, []string{testVLLMNativeSampler}),
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
			name: "exact CUDA 13 vllm-cpp",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeCUDA13,
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
			name: "vllm-cpp CUDA alias does not substitute CUDA 13",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeCUDA,
				Backends: []string{utils.BackendVLLMCpp},
			},
		},
		{
			name: "vllm-cpp exact CUDA 12 does not substitute CUDA 13",
			config: &config.InferenceConfig{
				Runtime:  utils.RuntimeCUDA12,
				Backends: []string{utils.BackendVLLMCpp},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ResolveBackend(test.config, platform)
			if err == nil {
				t.Fatal("ResolveBackend() succeeded, want error")
			}
			if !stderrors.Is(err, backendcatalog.ErrNotFound) {
				t.Fatalf("ResolveBackend() error = %v, want exact resolution failure", err)
			}
		})
	}
}

func TestResolveBackendReportsOmittedRuntimeAsCPU(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	_, err := ResolveBackend(&config.InferenceConfig{Backends: []string{"unknown"}}, platform)
	if err == nil {
		t.Fatal("ResolveBackend() succeeded, want error")
	}
	if !strings.Contains(err.Error(), `runtime "cpu"`) {
		t.Fatalf("ResolveBackend() error = %q, want normalized CPU runtime", err)
	}
	if strings.Contains(err.Error(), `runtime ""`) {
		t.Fatalf("ResolveBackend() error exposes an empty runtime: %q", err)
	}
}

func TestResolveBackendRejectsInvalidBackendLists(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	tests := []struct {
		name      string
		backends  []string
		wantError string
	}{
		{
			name:      "explicit empty family",
			backends:  []string{""},
			wantError: "backend cannot be empty",
		},
		{
			name:      "multiple families",
			backends:  []string{utils.BackendLlamaCpp, utils.BackendVLLM},
			wantError: "only one backend is supported at this time",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ResolveBackend(&config.InferenceConfig{Backends: test.backends}, platform)
			if err == nil {
				t.Fatal("ResolveBackend() succeeded, want error")
			}
			if !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("ResolveBackend() error = %v, want %q", err, test.wantError)
			}
		})
	}
}

func TestInstallBackendsUsesOnlyCatalogArtifacts(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)

	base := llb.Image(resolved.RuntimeBase.Ref, llb.Platform(platform))
	state := installBackends(resolved, backendcatalog.RuntimeCPU, platform, base, base)
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
				if metadata.URI != resolved.SourceRef || metadata.SourceRef != resolved.SourceRef || metadata.Runtime != string(backendcatalog.RuntimeCPU) || metadata.Status != string(resolved.Status) {
					t.Errorf("primary metadata = %+v, want selected entry provenance", metadata)
				}
			} else {
				if metadata.URI != metadata.Artifact {
					t.Errorf("fallback metadata URI = %q, want installed artifact %q", metadata.URI, metadata.Artifact)
				}
				if metadata.SourceRef != "" || metadata.Runtime != "" || metadata.Status != "" {
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
			metadataJSON := marshalBackendMetadata(resolved, backendcatalog.RuntimeCPU, test.artifact, test.primary)
			for iteration := 0; iteration < 10; iteration++ {
				if got := string(marshalBackendMetadata(resolved, backendcatalog.RuntimeCPU, test.artifact, test.primary)); got != string(metadataJSON) {
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
			if test.primary {
				if got := metadata["runtime"]; got != string(backendcatalog.RuntimeCPU) {
					t.Errorf("metadata runtime = %q, want %q", got, backendcatalog.RuntimeCPU)
				}
			} else if _, ok := metadata["runtime"]; ok {
				t.Error("fallback metadata unexpectedly contains runtime")
			}
			if _, ok := metadata["selector"]; ok {
				t.Error("metadata unexpectedly exposes the internal catalog selector")
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
	state := installBackends(resolved, backendcatalog.RuntimeCPU, platform, base, base)
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

	state := installBackends(resolved, backendcatalog.RuntimeCPU, platform, base, base)
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

	state := installBackends(resolved, backendcatalog.RuntimeCPU, platform, base, base)
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

func TestInstallBackendModelAliasesCoverPrimaryAndFallbackArtifacts(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)
	base := llb.Image(resolved.RuntimeBase.Ref, llb.Platform(platform))
	modelPaths := []string{testFasterWhisperModelPath, testRerankerModelPath}

	state := installBackends(resolved, backendcatalog.RuntimeCPU, platform, base, base)
	state = installBackendModelAliases(resolved, modelPaths, platform, state)
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal backend definition: %v", err)
	}

	aliasOp := findInferenceOpByCustomNamePrefix(t, definition, "Linking baked model directories into backend working directories")
	exec := aliasOp.op.GetExec()
	if exec == nil {
		t.Fatal("backend model alias operation is not an exec operation")
	}
	command := strings.Join(exec.Meta.Args, "\x00")

	artifacts := append([]backendcatalog.BackendArtifact{resolved.Backend}, resolved.Fallbacks...)
	for _, artifact := range artifacts {
		for _, modelPath := range modelPaths {
			for _, expected := range []string{
				"/aikit-root/backends/" + artifact.InstallName + "/" + modelPath,
				"/aikit-root/models/" + modelPath,
				"/models/" + modelPath,
			} {
				if !strings.Contains(command, expected) {
					t.Errorf("backend model alias command does not contain %q", expected)
				}
			}
		}
	}
	for _, guard := range []string{
		`require_real_directory "$backend_dir"`,
		`require_real_directory "$model_path"`,
		`ensure_real_directory "$ancestor"`,
		`[ -L "$alias_path" ]`,
		`actual_target=$(readlink "$alias_path")`,
		`[ "$actual_target" != "$model_target" ]`,
		`[ -e "$alias_path" ]`,
		`mkdir "$directory"`,
	} {
		if !strings.Contains(command, guard) {
			t.Errorf("backend model alias command does not contain guard %q", guard)
		}
	}
	mountedRoot := false
	for _, mount := range exec.Mounts {
		if mount.Dest == "/aikit-root" {
			mountedRoot = true
			break
		}
	}
	if !mountedRoot {
		t.Fatal("backend model alias operation does not mount the image root")
	}
}

func TestBackendModelAliasScriptCreatesExactAliasesIdempotently(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	resolved := testArbitraryBackendPlan(platform)
	modelPath := "organization/repository"
	imageRoot := t.TempDir()

	if err := os.MkdirAll(filepath.Join(imageRoot, "models", filepath.FromSlash(modelPath)), 0o755); err != nil {
		t.Fatalf("create baked model directory: %v", err)
	}
	artifacts := append([]backendcatalog.BackendArtifact{resolved.Backend}, resolved.Fallbacks...)
	for _, artifact := range artifacts {
		if err := os.MkdirAll(filepath.Join(imageRoot, "backends", artifact.InstallName), 0o755); err != nil {
			t.Fatalf("create backend directory %q: %v", artifact.InstallName, err)
		}
	}

	for iteration := 0; iteration < 2; iteration++ {
		output, err := executeBackendModelAliasScript(resolved, []string{modelPath}, imageRoot)
		if err != nil {
			t.Fatalf("execute backend model alias script iteration %d: %v: %s", iteration, err, output)
		}
	}

	wantTarget := "/models/" + modelPath
	for _, artifact := range artifacts {
		backendDir := filepath.Join(imageRoot, "backends", artifact.InstallName)
		ancestor := filepath.Join(backendDir, "organization")
		info, err := os.Lstat(ancestor)
		if err != nil {
			t.Fatalf("inspect alias ancestor for %q: %v", artifact.InstallName, err)
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			t.Fatalf("alias ancestor for %q has mode %s, want real directory", artifact.InstallName, info.Mode())
		}

		aliasPath := filepath.Join(backendDir, filepath.FromSlash(modelPath))
		info, err = os.Lstat(aliasPath)
		if err != nil {
			t.Fatalf("inspect alias for %q: %v", artifact.InstallName, err)
		}
		if info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("alias for %q has mode %s, want symlink", artifact.InstallName, info.Mode())
		}
		gotTarget, err := os.Readlink(aliasPath)
		if err != nil {
			t.Fatalf("read alias for %q: %v", artifact.InstallName, err)
		}
		if gotTarget != wantTarget {
			t.Fatalf("alias for %q points to %q, want %q", artifact.InstallName, gotTarget, wantTarget)
		}
	}
}

func TestBackendModelAliasScriptFailsClosedOnUnsafeLayouts(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	modelPath := "organization/repository"
	tests := []struct {
		name        string
		wantMessage string
		setup       func(t *testing.T, imageRoot string, backendDir string, modelDir string)
		verify      func(t *testing.T, imageRoot string, backendDir string)
	}{
		{
			name:        "missing baked model directory",
			wantMessage: "Backend model alias path is not a directory",
			setup: func(t *testing.T, _ string, backendDir string, _ string) {
				t.Helper()
				if err := os.MkdirAll(backendDir, 0o755); err != nil {
					t.Fatalf("create backend directory: %v", err)
				}
			},
		},
		{
			name:        "missing backend directory",
			wantMessage: "Backend model alias path is not a directory",
			setup: func(t *testing.T, _ string, _ string, modelDir string) {
				t.Helper()
				if err := os.MkdirAll(modelDir, 0o755); err != nil {
					t.Fatalf("create baked model directory: %v", err)
				}
			},
		},
		{
			name:        "backend directory symlink",
			wantMessage: "Backend model alias path is a symlink",
			setup: func(t *testing.T, imageRoot string, backendDir string, modelDir string) {
				t.Helper()
				if err := os.MkdirAll(modelDir, 0o755); err != nil {
					t.Fatalf("create baked model directory: %v", err)
				}
				outside := filepath.Join(imageRoot, "outside-backend")
				if err := os.MkdirAll(outside, 0o755); err != nil {
					t.Fatalf("create outside backend directory: %v", err)
				}
				if err := os.MkdirAll(filepath.Dir(backendDir), 0o755); err != nil {
					t.Fatalf("create backends directory: %v", err)
				}
				if err := os.Symlink(outside, backendDir); err != nil {
					t.Fatalf("create backend symlink: %v", err)
				}
			},
		},
		{
			name:        "alias ancestor symlink",
			wantMessage: "Backend model alias ancestor is a symlink",
			setup: func(t *testing.T, imageRoot string, backendDir string, modelDir string) {
				t.Helper()
				if err := os.MkdirAll(modelDir, 0o755); err != nil {
					t.Fatalf("create baked model directory: %v", err)
				}
				if err := os.MkdirAll(backendDir, 0o755); err != nil {
					t.Fatalf("create backend directory: %v", err)
				}
				outside := filepath.Join(imageRoot, "outside-ancestor")
				if err := os.MkdirAll(outside, 0o755); err != nil {
					t.Fatalf("create outside ancestor directory: %v", err)
				}
				if err := os.Symlink(outside, filepath.Join(backendDir, "organization")); err != nil {
					t.Fatalf("create ancestor symlink: %v", err)
				}
			},
			verify: func(t *testing.T, imageRoot string, _ string) {
				t.Helper()
				outsideAlias := filepath.Join(imageRoot, "outside-ancestor", "repository")
				if _, err := os.Lstat(outsideAlias); !os.IsNotExist(err) {
					t.Fatalf("outside alias unexpectedly exists or could not be inspected: %v", err)
				}
			},
		},
		{
			name:        "alias ancestor regular file",
			wantMessage: "Backend model alias ancestor is not a directory",
			setup: func(t *testing.T, _ string, backendDir string, modelDir string) {
				t.Helper()
				if err := os.MkdirAll(modelDir, 0o755); err != nil {
					t.Fatalf("create baked model directory: %v", err)
				}
				if err := os.MkdirAll(backendDir, 0o755); err != nil {
					t.Fatalf("create backend directory: %v", err)
				}
				if err := os.WriteFile(filepath.Join(backendDir, "organization"), []byte("conflict"), 0o600); err != nil {
					t.Fatalf("create ancestor file: %v", err)
				}
			},
		},
		{
			name:        "final alias regular file",
			wantMessage: "Backend model alias conflicts with existing path",
			setup: func(t *testing.T, _ string, backendDir string, modelDir string) {
				t.Helper()
				prepareBackendModelAliasParents(t, backendDir, modelDir)
				if err := os.WriteFile(filepath.Join(backendDir, filepath.FromSlash(modelPath)), []byte("conflict"), 0o600); err != nil {
					t.Fatalf("create alias file: %v", err)
				}
			},
		},
		{
			name:        "final alias directory",
			wantMessage: "Backend model alias conflicts with existing path",
			setup: func(t *testing.T, _ string, backendDir string, modelDir string) {
				t.Helper()
				prepareBackendModelAliasParents(t, backendDir, modelDir)
				if err := os.Mkdir(filepath.Join(backendDir, filepath.FromSlash(modelPath)), 0o755); err != nil {
					t.Fatalf("create alias directory: %v", err)
				}
			},
		},
		{
			name:        "final alias wrong symlink",
			wantMessage: "Backend model alias has unexpected target",
			setup: func(t *testing.T, _ string, backendDir string, modelDir string) {
				t.Helper()
				prepareBackendModelAliasParents(t, backendDir, modelDir)
				if err := os.Symlink("/models/wrong", filepath.Join(backendDir, filepath.FromSlash(modelPath))); err != nil {
					t.Fatalf("create wrong alias symlink: %v", err)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resolved := testArbitraryBackendPlan(platform)
			resolved.Fallbacks = nil
			imageRoot := t.TempDir()
			backendDir := filepath.Join(imageRoot, "backends", resolved.Backend.InstallName)
			modelDir := filepath.Join(imageRoot, "models", filepath.FromSlash(modelPath))
			test.setup(t, imageRoot, backendDir, modelDir)

			output, err := executeBackendModelAliasScript(resolved, []string{modelPath}, imageRoot)
			if err == nil {
				t.Fatalf("backend model alias script succeeded, want failure: %s", output)
			}
			if !strings.Contains(output, test.wantMessage) {
				t.Fatalf("backend model alias script output = %q, want message containing %q", output, test.wantMessage)
			}
			if test.verify != nil {
				test.verify(t, imageRoot, backendDir)
			}
		})
	}
}

func prepareBackendModelAliasParents(t *testing.T, backendDir string, modelDir string) {
	t.Helper()
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatalf("create baked model directory: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(backendDir, "organization"), 0o755); err != nil {
		t.Fatalf("create backend alias parent: %v", err)
	}
}

func executeBackendModelAliasScript(backend backendcatalog.Resolution, modelPaths []string, imageRoot string) (string, error) {
	command := exec.Command("sh")
	command.Stdin = strings.NewReader(backendModelAliasScript(backend, modelPaths, imageRoot))
	output, err := command.CombinedOutput()
	return string(output), err
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
