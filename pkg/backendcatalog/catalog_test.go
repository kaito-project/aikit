package backendcatalog

import (
	"crypto/sha256"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"regexp"
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
)

func TestDefaultResolvesCurrentRunnerTuples(t *testing.T) {
	t.Parallel()

	catalog, err := Default()
	if err != nil {
		t.Fatalf("parse default catalog: %v", err)
	}
	if got, want := len(catalog.Entries), 516; got != want {
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
		wantBase      Base
		wantFallbacks int
	}{
		{
			name: "diffusers CUDA 12 amd64",
			request: Request{
				Family: "diffusers", Selector: SelectorNVIDIA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA12,
			wantInstall: "cuda12-diffusers", wantRunner: RunnerProfileHFConfig, wantBase: BaseUbuntu,
		},
		{
			name: "llama CPU amd64",
			request: Request{
				Family: testFamilyLlamaCpp, Selector: SelectorDefault,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: testInstallCPULlama, wantRunner: RunnerProfileLlamaCpp, wantBase: BaseDistroless,
		},
		{
			name: "llama CPU arm64",
			request: Request{
				Family: testFamilyLlamaCpp, Selector: SelectorDefault,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: testInstallCPULlama, wantRunner: RunnerProfileLlamaCpp, wantBase: BaseDistroless,
		},
		{
			name: "llama CUDA 12 amd64",
			request: Request{
				Family: testFamilyLlamaCpp, Selector: SelectorNVIDIA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA12,
			wantInstall: testInstallCUDA12, wantRunner: RunnerProfileLlamaCpp, wantBase: BaseDistroless,
			wantFallbacks: 1,
		},
		{
			name: "vLLM CUDA 12 amd64",
			request: Request{
				Family: "vllm", Selector: SelectorNVIDIA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA12,
			wantInstall: "cuda12-vllm", wantRunner: RunnerProfileHFConfig, wantBase: BaseUbuntu,
		},
		{
			name: "vllm.cpp CPU amd64",
			request: Request{
				Family: testFamilyVLLMCpp, Selector: SelectorDefault,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: "cpu-vllm-cpp", wantRunner: RunnerProfileVLLMCpp, wantBase: BaseDistroless,
		},
		{
			name: "vllm.cpp CPU arm64",
			request: Request{
				Family: testFamilyVLLMCpp, Selector: SelectorDefault,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64},
			},
			wantRuntime: RuntimeCPU, wantTarget: TargetProfileCPU,
			wantInstall: "cpu-vllm-cpp", wantRunner: RunnerProfileVLLMCpp, wantBase: BaseDistroless,
		},
		{
			name: "vllm.cpp CUDA 13 amd64",
			request: Request{
				Family: testFamilyVLLMCpp, Selector: SelectorNVIDIA,
				Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64},
			},
			wantRuntime: RuntimeCUDA, wantTarget: TargetProfileCUDA13,
			wantInstall: "cuda13-vllm-cpp", wantRunner: RunnerProfileVLLMCpp, wantBase: BaseDistroless,
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
			if resolution.Base != tt.wantBase {
				t.Errorf("base = %q, want %q", resolution.Base, tt.wantBase)
			}
			if got := len(resolution.Fallbacks); got != tt.wantFallbacks {
				t.Errorf("fallback count = %d, want %d", got, tt.wantFallbacks)
			}
			if !strings.Contains(resolution.Core.Ref, "@sha256:") || !strings.Contains(resolution.Backend.Ref, "@sha256:") {
				t.Errorf("resolution contains mutable refs: core=%q backend=%q", resolution.Core.Ref, resolution.Backend.Ref)
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

	unknownRoot := strings.Replace(string(compact), `"schemaVersion":"v1"`, `"schemaVersion":"v1","unexpected":true`, 1)
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
		{name: "unknown schema", mutate: func(c *Catalog) { c.SchemaVersion = "v2" }},
		{name: "missing entries", mutate: func(c *Catalog) { c.Entries = nil }},
		{name: "invalid source", mutate: func(c *Catalog) { c.Source.Repository = "http://example.com/catalog" }},
		{name: "unsafe family", mutate: func(c *Catalog) { c.Entries[0].Family = "../llama-cpp" }},
		{name: "invalid selector", mutate: func(c *Catalog) { c.Entries[0].Selector = "cuda-automatic" }},
		{name: "invalid runtime", mutate: func(c *Catalog) { c.Entries[0].Runtime = "automatic" }},
		{name: "invalid target profile", mutate: func(c *Catalog) { c.Entries[0].TargetProfile = "cuda14" }},
		{name: "runtime target mismatch", mutate: func(c *Catalog) { c.Entries[0].TargetProfile = TargetProfileCPU }},
		{name: "selector target mismatch", mutate: func(c *Catalog) { c.Entries[0].Selector = SelectorCPU }},
		{name: "invalid status", mutate: func(c *Catalog) { c.Entries[0].Status = "stable" }},
		{name: "invalid channel", mutate: func(c *Catalog) { c.Entries[0].Channel = "nightly" }},
		{name: "invalid dependency profile", mutate: func(c *Catalog) { c.Entries[0].DependencyProfile = "automatic" }},
		{name: "missing runner profile", mutate: func(c *Catalog) { c.Entries[0].RunnerProfile = "" }},
		{name: "invalid base", mutate: func(c *Catalog) { c.Entries[0].Base = "debian" }},
		{name: "unsafe install name", mutate: func(c *Catalog) { c.Entries[0].Backend.InstallName = "../escape" }},
		{name: "mutable core tag", mutate: func(c *Catalog) { c.Entries[0].Core.Ref = "registry.example.com/localai/core:v1" }},
		{name: "backend tag and digest", mutate: func(c *Catalog) {
			c.Entries[0].Backend.Ref = "registry.example.com/localai/backend:v1@" + testDigestA
		}},
		{name: "uppercase digest", mutate: func(c *Catalog) {
			c.Entries[0].Backend.Ref = "registry.example.com/localai/backend@sha256:ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
		}},
		{name: "missing minimum CUDA", mutate: func(c *Catalog) { c.Entries[0].MinimumCUDA = "" }},
		{name: "distroless is not self contained", mutate: func(c *Catalog) {
			c.Entries[0].Base = BaseDistroless
			c.Entries[0].SelfContained = false
		}},
		{name: "duplicate workload", mutate: func(c *Catalog) { c.Entries[0].Workloads = []string{"chat", "chat"} }},
		{name: "unsafe fallback install name", mutate: func(c *Catalog) {
			c.Entries[0].Fallbacks[0].InstallName = "/tmp/backend"
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

func TestParseAcceptsExplicitTargetSelectors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		selector     Selector
		runtime      Runtime
		target       TargetProfile
		architecture string
		base         Base
	}{
		{name: "Intel", selector: SelectorIntel, runtime: RuntimeCPU, target: TargetProfileIntel},
		{name: "Vulkan", selector: SelectorVulkan, runtime: RuntimeAppleSilicon, target: TargetProfileVulkan, architecture: testArchitectureARM64, base: BaseAppleSilicon},
		{name: "Metal", selector: SelectorMetal, runtime: RuntimeAppleSilicon, target: TargetProfileMetal, architecture: testArchitectureARM64, base: BaseAppleSilicon},
		{name: "Metal Darwin arm64", selector: SelectorMetalDarwin, runtime: RuntimeAppleSilicon, target: TargetProfileMetal, architecture: testArchitectureARM64, base: BaseAppleSilicon},
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
			if tt.base != "" {
				entry.Base = tt.base
				entry.SelfContained = false
			}
			if tt.runtime != RuntimeCUDA {
				entry.MinimumCUDA = ""
			}

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
	entry.Base = BaseAppleSilicon
	entry.SelfContained = false
	entry.MinimumCUDA = ""

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
		{name: "Apple Silicon amd64", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeAppleSilicon
			entry.TargetProfile = TargetProfileVulkan
			entry.Selector = SelectorVulkan
			entry.Base = BaseAppleSilicon
			entry.SelfContained = false
			entry.MinimumCUDA = ""
		}},
		{name: "ROCm arm64", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeROCm
			entry.TargetProfile = TargetProfileROCm
			entry.Selector = SelectorAMD
			entry.Platform.Architecture = testArchitectureARM64
			entry.Base = BaseUbuntu24
			entry.SelfContained = false
			entry.MinimumCUDA = ""
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

func TestParseRejectsRuntimeBaseMismatches(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Entry)
	}{
		{name: "Apple Silicon requires its runtime base", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeAppleSilicon
			entry.TargetProfile = TargetProfileVulkan
			entry.Selector = SelectorVulkan
			entry.Platform.Architecture = testArchitectureARM64
			entry.Base = BaseUbuntu
			entry.SelfContained = false
			entry.MinimumCUDA = ""
		}},
		{name: "ROCm requires Ubuntu 24", mutate: func(entry *Entry) {
			entry.Runtime = RuntimeROCm
			entry.TargetProfile = TargetProfileROCm
			entry.Selector = SelectorAMD
			entry.Base = BaseUbuntu
			entry.SelfContained = false
			entry.MinimumCUDA = ""
		}},
		{name: "Apple Silicon base requires Apple runtime", mutate: func(entry *Entry) {
			entry.Base = BaseAppleSilicon
			entry.SelfContained = false
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
		Platform: Platform{OS: "linux", Architecture: "amd64"},
	}
	if _, err := resolver.Resolve(exact); err != nil {
		t.Fatalf("resolve exact tuple: %v", err)
	}

	notFound := []Request{
		{Family: testFamilyLlamaCpp, Selector: SelectorNVIDIA, Platform: exact.Platform},
		{Family: testFamilyLlamaCpp, Selector: exact.Selector, Platform: Platform{OS: testOSLinux, Architecture: testArchitectureARM64}},
		{Family: testFamilyLlamaCpp, Selector: exact.Selector, Platform: Platform{OS: testOSLinux, Architecture: testArchitectureAMD64, Variant: "v8"}},
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

	catalog.Entries[0].Backend.InstallName = "mutated-original"
	catalog.Entries[0].Fallbacks[0].InstallName = "mutated-fallback"
	catalog.Entries[0].Workloads[0] = "mutated-workload"

	first, err := resolver.Resolve(validTestRequest())
	if err != nil {
		t.Fatalf("first resolve: %v", err)
	}
	if first.Backend.InstallName != testInstallCUDA12 || first.Fallbacks[0].InstallName != testInstallCPULlama || first.Workloads[0] != testWorkloadText {
		t.Fatalf("resolver snapshot was mutated: %#v", first.Entry)
	}

	first.Fallbacks[0].InstallName = "mutated-result"
	first.Workloads[0] = "mutated-result"
	second, err := resolver.Resolve(validTestRequest())
	if err != nil {
		t.Fatalf("second resolve: %v", err)
	}
	if second.Fallbacks[0].InstallName != testInstallCPULlama || second.Workloads[0] != testWorkloadText {
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
		SchemaVersion: schemaVersionV1,
		Source: Source{
			Repository: "https://github.com/example/catalog",
			Revision:   "0123456789abcdef0123456789abcdef01234567",
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
				DependencyProfile: DependencyProfileNone,
				RunnerProfile:     RunnerProfileLlamaCpp,
				Base:              BaseDistroless,
				SelfContained:     true,
				MinimumCUDA:       "12.0",
				Workloads:         []string{testWorkloadText, "embeddings"},
			},
		},
	}
}

func validTestRequest() Request {
	return Request{
		Family:   testFamilyLlamaCpp,
		Selector: SelectorNVIDIACUDA12,
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
