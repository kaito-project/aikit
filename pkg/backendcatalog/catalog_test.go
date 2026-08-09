package backendcatalog

import (
	"crypto/sha256"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"testing"
)

const (
	testDigestA           = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	testDigestB           = "sha256:89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567"
	testOSLinux           = "linux"
	testArchitectureAMD64 = "amd64"
	testArchitectureARM64 = "arm64"
	testFamilyLlamaCpp    = string(RunnerProfileLlamaCpp)
	testFamilyVLLMCpp     = string(RunnerProfileVLLMCpp)
	testInstallCPULlama   = "cpu-llama-cpp"
	testInstallCUDA12     = "cuda12-llama-cpp"
	testWorkloadText      = "text-generation"
	testSystemPackageGCC  = "gcc"
	testInvalidRuntime    = "automatic"
	testMutatedResult     = "mutated-result"
	testSymlinkTarget     = "libcompat.so.1"
	testSymlinkPath       = "/usr/lib/libcompat.so.0"
	testCUDABuildType     = "BUILD_TYPE=cublas"
	testCUDAVisible       = "NVIDIA_VISIBLE_DEVICES=all"
	testVLLMNativeSampler = "VLLM_USE_FLASHINFER_SAMPLER=0"
	testUnsafePath        = "../escape"
)

var (
	testCUDA12Environment = []string{
		testCUDABuildType,
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_REQUIRE_CUDA=cuda>=12.0",
		testCUDAVisible,
	}
	testCUDA13Environment = []string{
		testCUDABuildType,
		"NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"NVIDIA_REQUIRE_CUDA=cuda>=13.0",
		testCUDAVisible,
	}
)

func TestDefaultResolvesCurrentRunnerTuples(t *testing.T) {
	t.Parallel()

	catalog, err := Default()
	if err != nil {
		t.Fatalf("parse default catalog: %v", err)
	}
	if got, want := len(catalog.Entries), 544; got != want {
		t.Fatalf("default entry count = %d, want %d", got, want)
	}
	if !regexp.MustCompile(`^sha256:[0-9a-f]{64}$`).MatchString(catalog.Digest()) {
		t.Fatalf("default digest = %q, want lowercase sha256 digest", catalog.Digest())
	}

	resolver, err := NewResolver(catalog)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	tests := []struct {
		name          string
		request       Request
		wantRuntime   Runtime
		wantTarget    TargetProfile
		wantInstall   string
		wantRunner    RunnerProfile
		wantPackages  []string
		wantEnv       []string
		wantFallbacks int
	}{
		{
			name: "diffusers CUDA 12 amd64",
			request: Request{
				Family: "diffusers", Selector: SelectorNVIDIA, Runtime: RuntimeCUDA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA12,
			wantInstall: "cuda12-diffusers", wantRunner: RunnerProfileHFConfig, wantEnv: testCUDA12Environment,
		},
		{
			name: "llama CPU amd64",
			request: Request{
				Family: testFamilyLlamaCpp, Selector: SelectorDefault, Runtime: RuntimeCPU,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: testInstallCPULlama, wantRunner: RunnerProfileLlamaCpp,
		},
		{
			name: "llama CPU arm64",
			request: Request{
				Family: testFamilyLlamaCpp, Selector: SelectorDefault, Runtime: RuntimeCPU,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: testInstallCPULlama, wantRunner: RunnerProfileLlamaCpp,
		},
		{
			name: "llama CUDA 12 amd64",
			request: Request{
				Family: testFamilyLlamaCpp, Selector: SelectorNVIDIA, Runtime: RuntimeCUDA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA12,
			wantInstall: testInstallCUDA12, wantRunner: RunnerProfileLlamaCpp, wantEnv: testCUDA12Environment,
			wantFallbacks: 1,
		},
		{
			name: "vLLM CUDA 12 amd64",
			request: Request{
				Family: "vllm", Selector: SelectorNVIDIA, Runtime: RuntimeCUDA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA12,
			wantInstall: "cuda12-vllm", wantRunner: RunnerProfileHFConfig,
			wantPackages: []string{testSystemPackageGCC, "libc6-dev"},
			wantEnv:      slices.Concat(testCUDA12Environment, []string{testVLLMNativeSampler}),
		},
		{
			name: "vllm.cpp CPU amd64",
			request: Request{
				Family: testFamilyVLLMCpp, Selector: SelectorDefault, Runtime: RuntimeCPU,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: "cpu-vllm-cpp", wantRunner: RunnerProfileVLLMCpp,
		},
		{
			name: "vllm.cpp CPU arm64",
			request: Request{
				Family: testFamilyVLLMCpp, Selector: SelectorDefault, Runtime: RuntimeCPU,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: "cpu-vllm-cpp", wantRunner: RunnerProfileVLLMCpp,
		},
		{
			name: "vllm.cpp CUDA 13 amd64",
			request: Request{
				Family: testFamilyVLLMCpp, Selector: SelectorNVIDIA, Runtime: RuntimeCUDA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA13,
			wantInstall: "cuda13-vllm-cpp", wantRunner: RunnerProfileVLLMCpp, wantEnv: testCUDA13Environment,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resolution, err := resolver.Resolve(tt.request)
			if err != nil {
				t.Fatalf("resolve tuple: %v", err)
			}
			if resolution.Runtime != tt.wantRuntime || resolution.TargetProfile != tt.wantTarget {
				t.Errorf("runtime/target = %q/%q, want %q/%q", resolution.Runtime, resolution.TargetProfile, tt.wantRuntime, tt.wantTarget)
			}
			if resolution.Backend.InstallName != tt.wantInstall {
				t.Errorf("install name = %q, want %q", resolution.Backend.InstallName, tt.wantInstall)
			}
			if resolution.RunnerProfile != tt.wantRunner {
				t.Errorf("runner profile = %q, want %q", resolution.RunnerProfile, tt.wantRunner)
			}
			if !slices.Equal(resolution.SystemPackages, tt.wantPackages) {
				t.Errorf("system packages = %q, want %q", resolution.SystemPackages, tt.wantPackages)
			}
			if !slices.Equal(resolution.Environment, tt.wantEnv) {
				t.Errorf("environment = %q, want %q", resolution.Environment, tt.wantEnv)
			}
			if got := len(resolution.Fallbacks); got != tt.wantFallbacks {
				t.Errorf("fallback count = %d, want %d", got, tt.wantFallbacks)
			}
			if !strings.Contains(resolution.RuntimeBase.Ref, "@sha256:") || !strings.Contains(resolution.Core.Ref, "@sha256:") || !strings.Contains(resolution.Backend.Ref, "@sha256:") {
				t.Errorf("resolution contains mutable refs: runtimeBase=%q core=%q backend=%q", resolution.RuntimeBase.Ref, resolution.Core.Ref, resolution.Backend.Ref)
			}
			if resolution.CatalogDigest != catalog.Digest() {
				t.Errorf("catalog digest = %q, want %q", resolution.CatalogDigest, catalog.Digest())
			}
			if resolution.Source != catalog.Source {
				t.Errorf("source = %#v, want %#v", resolution.Source, catalog.Source)
			}
		})
	}
}

func TestDefaultRejectsLiveQuarantinedTuples(t *testing.T) {
	t.Parallel()

	catalog, err := Default()
	if err != nil {
		t.Fatalf("parse default catalog: %v", err)
	}
	resolver, err := NewResolver(catalog)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	tests := []Request{
		{
			Family: "kokoro", Selector: SelectorDefault, Runtime: RuntimeCPU,
			Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
		},
		{
			Family: "sglang", Selector: SelectorNVIDIA, Runtime: RuntimeCUDA,
			Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
		},
		{
			Family: "sglang", Selector: Selector("nvidia-cuda-12"), Runtime: RuntimeCUDA,
			Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
		},
	}
	for _, request := range tests {
		if _, err := resolver.Resolve(request); !stderrors.Is(err, ErrUnavailable) {
			t.Errorf("resolve %s/%s error = %v, want ErrUnavailable", request.Family, request.Selector, err)
		}
	}
}

func TestDefaultCatalogRuntimePlanInvariants(t *testing.T) {
	t.Parallel()

	catalog, err := Default()
	if err != nil {
		t.Fatalf("parse default catalog: %v", err)
	}

	counts := make(map[TargetProfile]int)
	for _, entry := range catalog.Entries {
		counts[entry.TargetProfile]++
		switch entry.TargetProfile {
		case TargetProfileROCm:
			if !slices.Contains(entry.SystemPackages, "pciutils") {
				t.Errorf("ROCm entry %s/%s %s lacks pciutils", entry.Family, entry.Selector, formatPlatform(entry.Platform))
			}
			wantSymlink := RuntimeSymlink{Target: "libhipblaslt.so.1", Path: "/opt/rocm/lib/libhipblaslt.so.0"}
			if !slices.Contains(entry.RuntimeSymlinks, wantSymlink) {
				t.Errorf("ROCm entry %s/%s %s lacks compatibility symlink", entry.Family, entry.Selector, formatPlatform(entry.Platform))
			}
			if !strings.HasPrefix(entry.RuntimeBase.Ref, "docker.io/rocm/dev-ubuntu-24.04@sha256:") {
				t.Errorf("ROCm entry %s/%s %s runtime base = %q", entry.Family, entry.Selector, formatPlatform(entry.Platform), entry.RuntimeBase.Ref)
			}
		case TargetProfileL4TCUDA12, TargetProfileL4TCUDA13:
			minimumCUDA := "12.0"
			runtimeBasePrefix := "nvcr.io/nvidia/l4t-jetpack@sha256:"
			if entry.TargetProfile == TargetProfileL4TCUDA13 {
				minimumCUDA = "13.0"
				runtimeBasePrefix = "docker.io/library/ubuntu@sha256:"
			}
			wantEnvironment := []string{
				testCUDABuildType,
				"CUDA_HOME=/usr/local/cuda",
				"LD_LIBRARY_PATH=/usr/local/cuda/lib64:",
				"NVIDIA_DRIVER_CAPABILITIES=all",
				"NVIDIA_REQUIRE_CUDA=cuda>=" + minimumCUDA,
				testCUDAVisible,
				"PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			}
			if !slices.Equal(entry.Environment, wantEnvironment) {
				t.Errorf("L4T entry %s/%s %s environment = %q, want %q", entry.Family, entry.Selector, formatPlatform(entry.Platform), entry.Environment, wantEnvironment)
			}
			if !strings.HasPrefix(entry.RuntimeBase.Ref, runtimeBasePrefix) {
				t.Errorf("L4T entry %s/%s %s runtime base = %q", entry.Family, entry.Selector, formatPlatform(entry.Platform), entry.RuntimeBase.Ref)
			}
		case TargetProfileVulkan:
			switch entry.Platform.Architecture {
			case testArchitectureAMD64:
				if entry.Runtime != RuntimeCPU || !strings.HasPrefix(entry.RuntimeBase.Ref, "docker.io/library/ubuntu@sha256:") {
					t.Errorf("amd64 Vulkan entry %s/%s runtime/base = %q/%q", entry.Family, entry.Selector, entry.Runtime, entry.RuntimeBase.Ref)
				}
			case testArchitectureARM64:
				if entry.Runtime != RuntimeAppleSilicon || !strings.HasPrefix(entry.RuntimeBase.Ref, "ghcr.io/kaito-project/aikit/applesilicon/base@sha256:") {
					t.Errorf("arm64 Vulkan entry %s/%s runtime/base = %q/%q", entry.Family, entry.Selector, entry.Runtime, entry.RuntimeBase.Ref)
				}
				if !slices.Contains(entry.Environment, "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json") {
					t.Errorf("arm64 Vulkan entry %s/%s lacks Venus ICD environment", entry.Family, entry.Selector)
				}
			}
		}
	}

	for _, profile := range []TargetProfile{TargetProfileROCm, TargetProfileL4TCUDA12, TargetProfileL4TCUDA13, TargetProfileVulkan} {
		if counts[profile] == 0 {
			t.Errorf("default catalog has no %q runtime plans", profile)
		}
	}
}

func TestParseIsStrictAndDeterministic(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	compact := marshalCatalog(t, catalog)
	var document map[string]any
	if err := json.Unmarshal(compact, &document); err != nil {
		t.Fatalf("decode test document: %v", err)
	}
	indented, err := json.MarshalIndent(document, "", "    ")
	if err != nil {
		t.Fatalf("indent test document: %v", err)
	}

	parsedCompact, err := Parse(compact)
	if err != nil {
		t.Fatalf("parse compact catalog: %v", err)
	}
	parsedIndented, err := Parse(indented)
	if err != nil {
		t.Fatalf("parse indented catalog: %v", err)
	}
	if got, want := parsedCompact.Digest(), testBytesDigest(compact); got != want {
		t.Fatalf("compact digest = %q, want exact-byte digest %q", got, want)
	}
	if got, want := parsedIndented.Digest(), testBytesDigest(indented); got != want {
		t.Fatalf("indented digest = %q, want exact-byte digest %q", got, want)
	}
	if parsedCompact.Digest() == parsedIndented.Digest() {
		t.Fatal("differently encoded catalogs unexpectedly have the same exact-byte digest")
	}

	unknownRoot := strings.Replace(string(compact), `"schemaVersion":"v2"`, `"schemaVersion":"v2","unexpected":true`, 1)
	assertParseErrorIs(t, []byte(unknownRoot), ErrInvalidCatalog)

	unknownNested := strings.Replace(string(compact), `"platform":{"os":"linux"`, `"platform":{"unexpected":true,"os":"linux"`, 1)
	assertParseErrorIs(t, []byte(unknownNested), ErrInvalidCatalog)

	assertParseErrorIs(t, append(compact, []byte(` {}`)...), ErrInvalidCatalog)
}

func TestParseRejectsInvalidCatalogFields(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Catalog)
	}{
		{name: "unknown schema", mutate: func(c *Catalog) { c.SchemaVersion = "v3" }},
		{name: "missing entries", mutate: func(c *Catalog) { c.Entries = nil }},
		{name: "invalid source", mutate: func(c *Catalog) { c.Source.Repository = "http://example.com/catalog" }},
		{name: "unsafe family", mutate: func(c *Catalog) { c.Entries[0].Family = "../llama-cpp" }},
		{name: "invalid selector", mutate: func(c *Catalog) { c.Entries[0].Selector = "cuda-automatic" }},
		{name: "invalid runtime", mutate: func(c *Catalog) { c.Entries[0].Runtime = testInvalidRuntime }},
		{name: "invalid target profile", mutate: func(c *Catalog) { c.Entries[0].TargetProfile = "cuda14" }},
		{name: "runtime target mismatch", mutate: func(c *Catalog) { c.Entries[0].TargetProfile = TargetProfileCPU }},
		{name: "selector target mismatch", mutate: func(c *Catalog) { c.Entries[0].Selector = SelectorCPU }},
		{name: "invalid status", mutate: func(c *Catalog) { c.Entries[0].Status = "stable" }},
		{name: "invalid channel", mutate: func(c *Catalog) { c.Entries[0].Channel = "nightly" }},
		{name: "missing runner profile", mutate: func(c *Catalog) { c.Entries[0].RunnerProfile = "" }},
		{name: "unsafe install name", mutate: func(c *Catalog) { c.Entries[0].Backend.InstallName = testUnsafePath }},
		{name: "missing runtime base", mutate: func(c *Catalog) { c.Entries[0].RuntimeBase.Ref = "" }},
		{name: "mutable runtime base", mutate: func(c *Catalog) { c.Entries[0].RuntimeBase.Ref = "docker.io/library/ubuntu:24.04" }},
		{name: "mutable runner runtime base", mutate: func(c *Catalog) {
			c.Entries[0].RunnerRuntimeBase = &Artifact{Ref: "docker.io/library/ubuntu:22.04"}
		}},
		{name: "mutable core tag", mutate: func(c *Catalog) { c.Entries[0].Core.Ref = "registry.example.com/localai/core:v1" }},
		{name: "backend tag and digest", mutate: func(c *Catalog) {
			c.Entries[0].Backend.Ref = "registry.example.com/localai/backend:v1@" + testDigestA
		}},
		{name: "uppercase digest", mutate: func(c *Catalog) {
			c.Entries[0].Backend.Ref = "registry.example.com/localai/backend@sha256:ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
		}},
		{name: "unsafe system package", mutate: func(c *Catalog) { c.Entries[0].SystemPackages = []string{"gcc;curl"} }},
		{name: "duplicate system package", mutate: func(c *Catalog) {
			c.Entries[0].SystemPackages = []string{testSystemPackageGCC, testSystemPackageGCC}
		}},
		{name: "runner profile family mismatch", mutate: func(c *Catalog) { c.Entries[0].RunnerProfile = RunnerProfileVLLMCpp }},
		{name: "unsafe runtime symlink target", mutate: func(c *Catalog) { c.Entries[0].RuntimeSymlinks[0].Target = testUnsafePath }},
		{name: "relative runtime symlink path", mutate: func(c *Catalog) { c.Entries[0].RuntimeSymlinks[0].Path = "usr/lib/escape" }},
		{name: "duplicate runtime symlink path", mutate: func(c *Catalog) {
			c.Entries[0].RuntimeSymlinks = append(c.Entries[0].RuntimeSymlinks, c.Entries[0].RuntimeSymlinks[0])
		}},
		{name: "invalid environment name", mutate: func(c *Catalog) { c.Entries[0].Environment = []string{"lowercase=value"} }},
		{name: "environment without value separator", mutate: func(c *Catalog) { c.Entries[0].Environment = []string{"VALID_NAME"} }},
		{name: "multiline environment", mutate: func(c *Catalog) { c.Entries[0].Environment = []string{"VALID_NAME=value\nINJECTED=true"} }},
		{name: "duplicate environment name", mutate: func(c *Catalog) { c.Entries[0].Environment = []string{"VALID_NAME=one", "VALID_NAME=two"} }},
		{name: "duplicate workload", mutate: func(c *Catalog) { c.Entries[0].Workloads = []string{"chat", "chat"} }},
		{name: "unsafe fallback install name", mutate: func(c *Catalog) {
			c.Entries[0].Fallbacks[0].InstallName = "/tmp/backend"
		}},
		{name: "primary fallback install collision", mutate: func(c *Catalog) {
			c.Entries[0].Fallbacks[0].InstallName = c.Entries[0].Backend.InstallName
		}},
		{name: "fallback install collision", mutate: func(c *Catalog) {
			c.Entries[0].Fallbacks = append(c.Entries[0].Fallbacks, c.Entries[0].Fallbacks[0])
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			tt.mutate(&catalog)
			assertParseErrorIs(t, marshalCatalog(t, catalog), ErrInvalidCatalog)
		})
	}
}

func TestParseRejectsInvalidDefaults(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Defaults)
	}{
		{name: "missing family", mutate: func(defaults *Defaults) { defaults.Family = "" }},
		{name: "unsafe family", mutate: func(defaults *Defaults) { defaults.Family = testUnsafePath }},
		{name: "missing selectors", mutate: func(defaults *Defaults) { defaults.Selectors = nil }},
		{name: "invalid selector runtime", mutate: func(defaults *Defaults) { defaults.Selectors[0].Runtime = testInvalidRuntime }},
		{name: "invalid selector", mutate: func(defaults *Defaults) { defaults.Selectors[0].Selector = testInvalidRuntime }},
		{name: "duplicate runtime", mutate: func(defaults *Defaults) {
			defaults.Selectors = append(defaults.Selectors, defaults.Selectors[0])
		}},
		{name: "invalid platform override", mutate: func(defaults *Defaults) {
			defaults.Selectors = append(defaults.Selectors, DefaultSelector{
				Runtime: RuntimeCUDA, Platform: &Platform{OS: testOSLinux, Architecture: testArchitectureARM64, Variant: "v8"}, Selector: SelectorNVIDIAL4T,
			})
		}},
		{name: "duplicate platform override", mutate: func(defaults *Defaults) {
			platform := Platform{OS: testOSLinux, Architecture: testArchitectureARM64}
			defaults.Selectors = append(defaults.Selectors,
				DefaultSelector{Runtime: RuntimeCUDA, Platform: &platform, Selector: SelectorNVIDIAL4T},
				DefaultSelector{Runtime: RuntimeCUDA, Platform: &platform, Selector: SelectorL4TCUDA12},
			)
		}},
		{name: "missing runtime", mutate: func(defaults *Defaults) {
			defaults.Selectors = defaults.Selectors[:len(defaults.Selectors)-1]
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			tt.mutate(&catalog.Defaults)
			assertParseErrorIs(t, marshalCatalog(t, catalog), ErrInvalidCatalog)
		})
	}
}

func TestParseAcceptsExplicitTargetSelectors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		selector     Selector
		runtime      Runtime
		target       TargetProfile
		architecture string
	}{
		{name: "Intel", selector: SelectorIntel, runtime: RuntimeCPU, target: TargetProfileIntel},
		{name: "Vulkan amd64", selector: SelectorVulkan, runtime: RuntimeCPU, target: TargetProfileVulkan},
		{name: "Vulkan arm64", selector: SelectorVulkan, runtime: RuntimeAppleSilicon, target: TargetProfileVulkan, architecture: testArchitectureARM64},
		{name: "Metal", selector: SelectorMetal, runtime: RuntimeAppleSilicon, target: TargetProfileMetal, architecture: testArchitectureARM64},
		{name: "Metal Darwin arm64", selector: SelectorMetalDarwin, runtime: RuntimeAppleSilicon, target: TargetProfileMetal, architecture: testArchitectureARM64},
		{name: "generic L4T CUDA 12", selector: SelectorNVIDIAL4T, runtime: RuntimeCUDA, target: TargetProfileL4TCUDA12, architecture: testArchitectureARM64},
		{name: "exact L4T CUDA 12", selector: SelectorL4TCUDA12, runtime: RuntimeCUDA, target: TargetProfileL4TCUDA12, architecture: testArchitectureARM64},
		{name: "exact L4T CUDA 13", selector: SelectorL4TCUDA13, runtime: RuntimeCUDA, target: TargetProfileL4TCUDA13, architecture: testArchitectureARM64},
		{name: "open L4T CUDA 12", selector: "nvidia-l4t-cuda-12-jetpack-6", runtime: RuntimeCUDA, target: TargetProfileL4TCUDA12, architecture: testArchitectureARM64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			entry := &catalog.Entries[0]
			entry.Selector = tt.selector
			entry.Runtime = tt.runtime
			entry.TargetProfile = tt.target
			if tt.architecture != "" {
				entry.Platform.Architecture = tt.architecture
			}
			entry.SystemPackages = nil
			entry.Environment = nil

			if _, err := Parse(marshalCatalog(t, catalog)); err != nil {
				t.Fatalf("parse catalog with selector %q: %v", tt.selector, err)
			}
		})
	}
}

func TestParseAllowsExperimentalDefaultSelectorForCUDA(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	catalog.Entries[0].Selector = SelectorDefault
	catalog.Entries[0].Status = StatusExperimental

	if _, err := Parse(marshalCatalog(t, catalog)); err != nil {
		t.Fatalf("parse experimental default selector for CUDA target: %v", err)
	}
}

func TestParseAllowsExperimentalIntelSelectorForVulkan(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	entry := &catalog.Entries[0]
	entry.Selector = SelectorIntel
	entry.Runtime = RuntimeAppleSilicon
	entry.TargetProfile = TargetProfileVulkan
	entry.Status = StatusExperimental
	entry.Platform.Architecture = testArchitectureARM64
	entry.SystemPackages = nil
	entry.Environment = nil

	if _, err := Parse(marshalCatalog(t, catalog)); err != nil {
		t.Fatalf("parse experimental Intel selector for Vulkan target: %v", err)
	}
}

func TestParseRejectsUnsupportedRuntimePlatforms(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Entry)
	}{
		{name: "non-Linux", mutate: func(entry *Entry) { entry.Platform.OS = "darwin" }},
		{name: "unsupported architecture", mutate: func(entry *Entry) { entry.Platform.Architecture = "ppc64le" }},
		{name: "noncanonical ARM64 variant", mutate: func(entry *Entry) {
			entry.Platform.Architecture = testArchitectureARM64
			entry.Platform.Variant = "v8"
		}},
		{name: "Apple Silicon amd64", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeAppleSilicon
			entry.TargetProfile = TargetProfileVulkan
			entry.Selector = SelectorVulkan
			entry.SystemPackages = nil
			entry.Environment = nil
		}},
		{name: "CPU Vulkan arm64", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeCPU
			entry.TargetProfile = TargetProfileVulkan
			entry.Selector = SelectorVulkan
			entry.Platform.Architecture = testArchitectureARM64
			entry.SystemPackages = nil
			entry.Environment = nil
		}},
		{name: "ROCm arm64", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeROCm
			entry.TargetProfile = TargetProfileROCm
			entry.Selector = SelectorAMD
			entry.Platform.Architecture = testArchitectureARM64
			entry.SystemPackages = nil
			entry.Environment = nil
		}},
		{name: "L4T amd64", mutate: func(entry *Entry) {
			entry.TargetProfile = TargetProfileL4TCUDA12
			entry.Selector = SelectorL4TCUDA12
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			test.mutate(&catalog.Entries[0])
			assertParseErrorIs(t, marshalCatalog(t, catalog), ErrInvalidCatalog)
		})
	}
}

func TestParseRejectsInvalidRuntimeBase(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Entry)
	}{
		{name: "missing", mutate: func(entry *Entry) { entry.RuntimeBase.Ref = "" }},
		{name: "mutable tag", mutate: func(entry *Entry) { entry.RuntimeBase.Ref = "docker.io/library/ubuntu:24.04" }},
		{name: "tag and digest", mutate: func(entry *Entry) {
			entry.RuntimeBase.Ref = "docker.io/library/ubuntu:24.04@" + testDigestA
		}},
		{name: "uppercase digest", mutate: func(entry *Entry) {
			entry.RuntimeBase.Ref = "docker.io/library/ubuntu@sha256:ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			test.mutate(&catalog.Entries[0])
			assertParseErrorIs(t, marshalCatalog(t, catalog), ErrInvalidCatalog)
		})
	}
}

func TestParseRejectsSupportedSelectorTargetMismatch(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	catalog.Entries[0].Selector = SelectorDefault
	catalog.Entries[0].Status = StatusSupported

	_, err := Parse(marshalCatalog(t, catalog))
	if !stderrors.Is(err, ErrInvalidCatalog) {
		t.Fatalf("Parse() error = %v, want ErrInvalidCatalog", err)
	}
	if !strings.Contains(err.Error(), "does not match targetProfile") {
		t.Fatalf("Parse() error = %v, want selector/target mismatch", err)
	}
}

func TestParseRejectsDuplicateTuple(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	duplicate := cloneEntry(catalog.Entries[0])
	duplicate.Version = "v9.9.9"
	duplicate.Backend.Ref = "registry.example.com/localai/backend@" + testDigestB
	catalog.Entries = append(catalog.Entries, duplicate)

	assertParseErrorIs(t, marshalCatalog(t, catalog), ErrInvalidCatalog)
}

func TestResolverAppliesCatalogDefaultsForRuntime(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	cpuEntry := cloneEntry(catalog.Entries[0])
	cpuEntry.Selector = SelectorDefault
	cpuEntry.Runtime = RuntimeCPU
	cpuEntry.TargetProfile = TargetProfileCPU
	cpuEntry.Backend.InstallName = testInstallCPULlama
	cpuEntry.Fallbacks = nil
	cpuEntry.SystemPackages = nil
	cpuEntry.Environment = nil
	catalog.Entries = append(catalog.Entries, cpuEntry)

	parsed, err := Parse(marshalCatalog(t, catalog))
	if err != nil {
		t.Fatalf("parse catalog: %v", err)
	}
	resolver, err := NewResolver(parsed)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	tests := []struct {
		name         string
		runtime      Runtime
		wantSelector Selector
		wantInstall  string
	}{
		{name: "CPU", runtime: RuntimeCPU, wantSelector: SelectorDefault, wantInstall: testInstallCPULlama},
		{name: "CUDA", runtime: RuntimeCUDA, wantSelector: SelectorNVIDIACUDA12, wantInstall: testInstallCUDA12},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resolution, err := resolver.Resolve(Request{
				Runtime:  tt.runtime,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			})
			if err != nil {
				t.Fatalf("resolve defaults: %v", err)
			}
			if resolution.Family != catalog.Defaults.Family || resolution.Selector != tt.wantSelector || resolution.Backend.InstallName != tt.wantInstall {
				t.Fatalf("default resolution = %q/%q/%q, want %q/%q/%q", resolution.Family, resolution.Selector, resolution.Backend.InstallName, catalog.Defaults.Family, tt.wantSelector, tt.wantInstall)
			}
		})
	}
}

func TestResolverPrefersPlatformScopedDefault(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	for i := range catalog.Defaults.Selectors {
		if catalog.Defaults.Selectors[i].Runtime == RuntimeCUDA {
			catalog.Defaults.Selectors[i].Selector = SelectorNVIDIA
		}
	}
	arm64 := Platform{OS: testOSLinux, Architecture: testArchitectureARM64}
	catalog.Defaults.Selectors = append(catalog.Defaults.Selectors, DefaultSelector{
		Runtime: RuntimeCUDA, Platform: &arm64, Selector: SelectorNVIDIAL4T,
	})
	catalog.Entries[0].Selector = SelectorNVIDIA
	arm64Entry := cloneEntry(catalog.Entries[0])
	arm64Entry.Selector = SelectorNVIDIAL4T
	arm64Entry.Platform = arm64
	arm64Entry.TargetProfile = TargetProfileL4TCUDA12
	arm64Entry.Backend.InstallName = "l4t-llama-cpp"
	catalog.Entries = append(catalog.Entries, arm64Entry)

	parsed, err := Parse(marshalCatalog(t, catalog))
	if err != nil {
		t.Fatalf("parse catalog: %v", err)
	}
	resolver, err := NewResolver(parsed)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	tests := []struct {
		name         string
		request      Request
		wantSelector Selector
		wantErr      error
	}{
		{
			name: "generic amd64 default",
			request: Request{Runtime: RuntimeCUDA, Platform: Platform{
				OS: testOSLinux, Architecture: testArchitectureAMD64,
			}},
			wantSelector: SelectorNVIDIA,
		},
		{
			name: "generic amd64 default with compatible variant",
			request: Request{Runtime: RuntimeCUDA, Platform: Platform{
				OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v3",
			}},
			wantSelector: SelectorNVIDIA,
		},
		{
			name: "platform arm64 default",
			request: Request{Runtime: RuntimeCUDA, Platform: Platform{
				OS: testOSLinux, Architecture: testArchitectureARM64,
			}},
			wantSelector: SelectorNVIDIAL4T,
		},
		{
			name: "platform arm64 default with compatible variant",
			request: Request{Runtime: RuntimeCUDA, Platform: Platform{
				OS: testOSLinux, Architecture: testArchitectureARM64, Variant: "v8",
			}},
			wantSelector: SelectorNVIDIAL4T,
		},
		{
			name: "explicit selector does not use platform default",
			request: Request{Selector: SelectorNVIDIA, Runtime: RuntimeCUDA, Platform: Platform{
				OS: testOSLinux, Architecture: testArchitectureARM64,
			}},
			wantErr: ErrNotFound,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resolution, resolveErr := resolver.Resolve(tt.request)
			if tt.wantErr != nil {
				if !stderrors.Is(resolveErr, tt.wantErr) {
					t.Fatalf("Resolve() error = %v, want %v", resolveErr, tt.wantErr)
				}
				return
			}
			if resolveErr != nil {
				t.Fatalf("Resolve() error = %v", resolveErr)
			}
			if resolution.Selector != tt.wantSelector {
				t.Errorf("Resolve() selector = %q, want %q", resolution.Selector, tt.wantSelector)
			}
		})
	}
}

func TestResolverMatchesExactTupleAndFailsClosed(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	resolver, err := NewResolver(&catalog)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	exact := Request{
		Family:   "llama-cpp",
		Selector: SelectorNVIDIACUDA12,
		Runtime:  RuntimeCUDA,
		Platform: Platform{OS: "linux", Architecture: "amd64"},
	}
	if _, err := resolver.Resolve(exact); err != nil {
		t.Fatalf("resolve exact tuple: %v", err)
	}

	notFound := []Request{
		{Family: testFamilyLlamaCpp, Selector: SelectorNVIDIA, Runtime: exact.Runtime, Platform: exact.Platform},
		{Family: testFamilyLlamaCpp, Selector: exact.Selector, Runtime: exact.Runtime, Platform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64}},
		{Family: testFamilyLlamaCpp, Selector: exact.Selector, Runtime: exact.Runtime, Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v8"}},
		{Family: testFamilyLlamaCpp, Selector: exact.Selector, Runtime: RuntimeCPU, Platform: exact.Platform},
	}
	for _, request := range notFound {
		if _, err := resolver.Resolve(request); !stderrors.Is(err, ErrNotFound) {
			t.Errorf("resolve non-exact tuple error = %v, want ErrNotFound", err)
		}
	}

	invalid := exact
	invalid.Family = "../llama-cpp"
	if _, err := resolver.Resolve(invalid); !stderrors.Is(err, ErrInvalidRequest) {
		t.Fatalf("resolve invalid request error = %v, want ErrInvalidRequest", err)
	}

	invalid = exact
	invalid.Runtime = testInvalidRuntime
	if _, err := resolver.Resolve(invalid); !stderrors.Is(err, ErrInvalidRequest) {
		t.Fatalf("resolve invalid runtime error = %v, want ErrInvalidRequest", err)
	}
}

func TestResolverNormalizesPlatformAliases(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	catalog.Entries[0].Platform = Platform{OS: testOSLinux, Architecture: testArchitectureARM64}
	resolver, err := NewResolver(&catalog)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	for _, platform := range []Platform{
		{OS: "LINUX", Architecture: "aarch64"},
		{OS: testOSLinux, Architecture: testArchitectureARM64, Variant: "v8"},
	} {
		request := validTestRequest()
		request.Platform = platform
		if _, err := resolver.Resolve(request); err != nil {
			t.Errorf("resolve platform alias %#v: %v", platform, err)
		}
	}
}

func TestResolverIgnoresCompatibleCPUVariantsForLookup(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                  string
		entryPlatform         Platform
		compatiblePlatforms   []Platform
		incompatiblePlatforms []Platform
	}{
		{
			name:          "linux amd64",
			entryPlatform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			compatiblePlatforms: []Platform{
				{OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v2"},
				{OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v3"},
				{OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v4"},
			},
			incompatiblePlatforms: []Platform{
				{OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v5"},
				{OS: "darwin", Architecture: testArchitectureAMD64, Variant: "v3"},
			},
		},
		{
			name:          "linux arm64",
			entryPlatform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64},
			compatiblePlatforms: []Platform{
				{OS: testOSLinux, Architecture: testArchitectureARM64, Variant: "v8"},
			},
			incompatiblePlatforms: []Platform{
				{OS: testOSLinux, Architecture: testArchitectureARM64, Variant: "v7"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			catalog.Entries[0].Platform = tt.entryPlatform
			resolver, err := NewResolver(&catalog)
			if err != nil {
				t.Fatalf("create resolver: %v", err)
			}

			for _, platform := range tt.compatiblePlatforms {
				request := validTestRequest()
				request.Platform = platform
				if _, err := resolver.Resolve(request); err != nil {
					t.Errorf("resolve compatible platform %#v: %v", platform, err)
				}
			}
			for _, platform := range tt.incompatiblePlatforms {
				request := validTestRequest()
				request.Platform = platform
				if _, err := resolver.Resolve(request); !stderrors.Is(err, ErrNotFound) {
					t.Errorf("resolve incompatible platform %#v error = %v, want ErrNotFound", platform, err)
				}
			}
		})
	}
}

func TestResolverRejectsUnavailableStatuses(t *testing.T) {
	t.Parallel()

	for _, status := range []Status{StatusQuarantined, StatusDeprecated} {
		t.Run(string(status), func(t *testing.T) {
			t.Parallel()

			catalog := validTestCatalog()
			catalog.Entries[0].Status = status
			resolver, err := NewResolver(&catalog)
			if err != nil {
				t.Fatalf("create resolver: %v", err)
			}
			if _, err := resolver.Resolve(validTestRequest()); !stderrors.Is(err, ErrUnavailable) {
				t.Fatalf("resolve status %q error = %v, want ErrUnavailable", status, err)
			}
		})
	}

	catalog := validTestCatalog()
	catalog.Entries[0].Status = StatusExperimental
	resolver, err := NewResolver(&catalog)
	if err != nil {
		t.Fatalf("create experimental resolver: %v", err)
	}
	if _, err := resolver.Resolve(validTestRequest()); err != nil {
		t.Fatalf("resolve experimental entry: %v", err)
	}
}

func TestResolverSnapshotsCatalogAndDetachesResults(t *testing.T) {
	t.Parallel()

	catalog := validTestCatalog()
	resolver, err := NewResolver(&catalog)
	if err != nil {
		t.Fatalf("create resolver: %v", err)
	}

	catalog.Defaults.Family = "mutated-family"
	catalog.Defaults.Selectors[1].Selector = SelectorNVIDIA
	catalog.Entries[0].Backend.InstallName = "mutated-original"
	catalog.Entries[0].Fallbacks[0].InstallName = "mutated-fallback"
	catalog.Entries[0].SystemPackages[0] = "mutated-package"
	catalog.Entries[0].RuntimeSymlinks[0].Path = "/mutated"
	catalog.Entries[0].Environment[0] = "MUTATED=original"
	catalog.Entries[0].Workloads[0] = "mutated-workload"

	defaultRequest := Request{
		Runtime:  RuntimeCUDA,
		Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
	}
	first, err := resolver.Resolve(defaultRequest)
	if err != nil {
		t.Fatalf("first resolve: %v", err)
	}
	if first.Family != testFamilyLlamaCpp || first.Selector != SelectorNVIDIACUDA12 || first.Backend.InstallName != testInstallCUDA12 ||
		first.Fallbacks[0].InstallName != testInstallCPULlama || first.SystemPackages[0] != testSystemPackageGCC ||
		first.RuntimeSymlinks[0].Path != testSymlinkPath || first.Environment[0] != testCUDA12Environment[0] || first.Workloads[0] != testWorkloadText {
		t.Fatalf("resolver snapshot was mutated: %#v", first.Entry)
	}

	first.Fallbacks[0].InstallName = testMutatedResult
	first.SystemPackages[0] = testMutatedResult
	first.RuntimeSymlinks[0].Path = "/mutated-result"
	first.Environment[0] = "MUTATED=result"
	first.Workloads[0] = testMutatedResult
	second, err := resolver.Resolve(defaultRequest)
	if err != nil {
		t.Fatalf("second resolve: %v", err)
	}
	if second.Fallbacks[0].InstallName != testInstallCPULlama || second.SystemPackages[0] != testSystemPackageGCC ||
		second.RuntimeSymlinks[0].Path != testSymlinkPath || second.Environment[0] != testCUDA12Environment[0] || second.Workloads[0] != testWorkloadText {
		t.Fatalf("returned resolution mutated resolver state: %#v", second.Entry)
	}
}

func TestDefaultReturnsFreshCatalog(t *testing.T) {
	t.Parallel()

	first, err := Default()
	if err != nil {
		t.Fatalf("first default catalog: %v", err)
	}
	first.Entries[0].Family = "mutated"

	second, err := Default()
	if err != nil {
		t.Fatalf("second default catalog: %v", err)
	}
	if second.Entries[0].Family == "mutated" {
		t.Fatal("Default returned shared mutable catalog state")
	}
}

func validTestCatalog() Catalog {
	return Catalog{
		SchemaVersion: schemaVersionV2,
		Source: Source{
			Repository: "https://github.com/example/catalog",
			Revision:   "0123456789abcdef0123456789abcdef01234567",
		},
		Defaults: Defaults{
			Family: testFamilyLlamaCpp,
			Selectors: []DefaultSelector{
				{Runtime: RuntimeCPU, Selector: SelectorDefault},
				{Runtime: RuntimeCUDA, Selector: SelectorNVIDIACUDA12},
				{Runtime: RuntimeROCm, Selector: SelectorAMD},
				{Runtime: RuntimeAppleSilicon, Selector: SelectorVulkan},
			},
		},
		Entries: []Entry{
			{
				Family:        testFamilyLlamaCpp,
				Selector:      SelectorNVIDIACUDA12,
				Platform:      Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
				Runtime:       RuntimeCUDA,
				TargetProfile: TargetProfileCUDA12,
				Status:        StatusSupported,
				Channel:       ChannelStable,
				Version:       "v1.2.3",
				SourceRef:     "registry.example.com/localai/backend:v1.2.3-cuda12-llama-cpp",
				RuntimeBase: Artifact{
					Ref: "docker.io/library/ubuntu@" + testDigestA,
				},
				Core: Artifact{
					Ref: "registry.example.com/localai/core@" + testDigestA,
				},
				Backend: BackendArtifact{
					Ref:         "registry.example.com/localai/backend@" + testDigestB,
					InstallName: testInstallCUDA12,
				},
				Fallbacks: []BackendArtifact{
					{
						Ref:         "registry.example.com/localai/backend@" + testDigestA,
						InstallName: testInstallCPULlama,
					},
				},
				SystemPackages:  []string{testSystemPackageGCC, "libc6-dev"},
				RuntimeSymlinks: []RuntimeSymlink{{Target: testSymlinkTarget, Path: testSymlinkPath}},
				Environment:     append([]string(nil), testCUDA12Environment...),
				RunnerProfile:   RunnerProfileLlamaCpp,
				Workloads:       []string{testWorkloadText, "embeddings"},
			},
		},
	}
}

func validTestRequest() Request {
	return Request{
		Family:   testFamilyLlamaCpp,
		Selector: SelectorNVIDIACUDA12,
		Runtime:  RuntimeCUDA,
		Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
	}
}

func marshalCatalog(t *testing.T, catalog Catalog) []byte {
	t.Helper()

	data, err := json.Marshal(catalog)
	if err != nil {
		t.Fatalf("marshal test catalog: %v", err)
	}

	return data
}

func assertParseErrorIs(t *testing.T, data []byte, target error) {
	t.Helper()

	if _, err := Parse(data); !stderrors.Is(err, target) {
		t.Fatalf("Parse() error = %v, want %v", err, target)
	}
}

func testBytesDigest(data []byte) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256(data))
}
