package backendcatalogimport

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/backendcatalog"
)

func TestGenerateDeterministicCatalog(t *testing.T) {
	source := readFixture(t, "testdata/index.yaml")
	resolver := readSnapshot(t, "testdata/resolutions.json")
	options := GenerateOptions{
		Source:          fixturePin(source),
		Version:         LocalAIVersion,
		CoreRefTemplate: fixtureCoreRefTemplate,
		Resolver:        resolver,
	}

	first, err := Generate(context.Background(), source, options)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	second, err := Generate(context.Background(), source, options)
	if err != nil {
		t.Fatalf("Generate() second error = %v", err)
	}
	firstJSON, err := Marshal(first)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	secondJSON, err := Marshal(second)
	if err != nil {
		t.Fatalf("Marshal() second error = %v", err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatal("Generate() output is not deterministic")
	}
	if _, err := backendcatalog.Parse(firstJSON); err != nil {
		t.Fatalf("generated output does not parse as pkg/backendcatalog v2: %v\n%s", err, firstJSON)
	}
	if first.SchemaVersion != SchemaVersion {
		t.Fatalf("schemaVersion = %q, want %q", first.SchemaVersion, SchemaVersion)
	}
	if first.Defaults.Family != defaultFamily {
		t.Fatalf("defaults.family = %q, want %q", first.Defaults.Family, defaultFamily)
	}
	wantDefaults := []DefaultSelector{
		{Runtime: runtimeApple, Selector: targetVulkan},
		{Runtime: runtimeCPU, Selector: selectorDefault},
		{Runtime: runtimeCUDA, Selector: selectorNVIDIA},
		{
			Runtime: runtimeCUDA,
			Platform: &Platform{
				OS:           platformLinux,
				Architecture: architectureARM64,
			},
			Selector: selectorNVIDIAL4T,
		},
		{Runtime: runtimeROCm, Selector: selectorAMD},
	}
	if len(first.Defaults.Selectors) != len(wantDefaults) {
		t.Fatalf("defaults.selectors = %#v, want %#v", first.Defaults.Selectors, wantDefaults)
	}
	for index, want := range wantDefaults {
		if !reflect.DeepEqual(first.Defaults.Selectors[index], want) {
			t.Fatalf("defaults.selectors[%d] = %#v, want %#v", index, first.Defaults.Selectors[index], want)
		}
	}

	if len(first.Entries) != 6 {
		t.Fatalf("entry count = %d, want 6", len(first.Entries))
	}
	amd64CPU := findGeneratedEntry(t, first, runnerLlamaCpp, selectorDefault, Platform{OS: platformLinux, Architecture: architectureAMD64})
	if amd64CPU.Backend.Ref != "registry.example/local-ai-backends@"+fixtureDigestA {
		t.Errorf("CPU backend ref = %q", amd64CPU.Backend.Ref)
	}
	if amd64CPU.Core.Ref != "registry.example/core@sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" {
		t.Errorf("CPU core ref = %q", amd64CPU.Core.Ref)
	}
	if amd64CPU.RuntimeBase.Ref != "ghcr.io/kaito-project/aikit/base@"+fixtureChiseledAMD64 {
		t.Errorf("CPU runtime base ref = %q", amd64CPU.RuntimeBase.Ref)
	}
	if amd64CPU.RunnerRuntimeBase == nil || amd64CPU.RunnerRuntimeBase.Ref != "docker.io/library/ubuntu@"+fixtureUbuntu22AMD64 {
		t.Errorf("CPU runner runtime base = %#v", amd64CPU.RunnerRuntimeBase)
	}
	if len(amd64CPU.SystemPackages) != 0 || len(amd64CPU.Environment) != 0 {
		t.Errorf("CPU packages/environment = %v/%v, want empty", amd64CPU.SystemPackages, amd64CPU.Environment)
	}
	if got, want := strings.Join(amd64CPU.Workloads, ","), "cpu,llm,text-to-text"; got != want {
		t.Errorf("workloads = %q, want %q", got, want)
	}
	arm64CPU := findGeneratedEntry(t, first, runnerLlamaCpp, selectorDefault, Platform{OS: platformLinux, Architecture: architectureARM64})
	if arm64CPU.Platform != (Platform{OS: platformLinux, Architecture: architectureARM64}) {
		t.Fatalf("second entry platform = %#v, want normalized linux/arm64", arm64CPU.Platform)
	}
	if arm64CPU.Backend.Ref != "registry.example/local-ai-backends@"+fixtureDigestB {
		t.Errorf("arm64 backend ref = %q", arm64CPU.Backend.Ref)
	}
	if arm64CPU.Core.Ref != "registry.example/core@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" {
		t.Errorf("arm64 core ref = %q", arm64CPU.Core.Ref)
	}
	if arm64CPU.RuntimeBase.Ref != "ghcr.io/kaito-project/aikit/base@"+fixtureChiseledARM64 {
		t.Errorf("arm64 runtime base ref = %q", arm64CPU.RuntimeBase.Ref)
	}
	if arm64CPU.RunnerRuntimeBase == nil || arm64CPU.RunnerRuntimeBase.Ref != "docker.io/library/ubuntu@"+fixtureUbuntu22ARM64 {
		t.Errorf("arm64 runner runtime base = %#v", arm64CPU.RunnerRuntimeBase)
	}

	nvidia := findGeneratedEntry(t, first, runnerLlamaCpp, selectorNVIDIA, Platform{OS: platformLinux, Architecture: architectureAMD64})
	if nvidia.Selector != selectorNVIDIA || nvidia.TargetProfile != targetCUDA12 {
		t.Fatalf("NVIDIA policy = selector %q target %q", nvidia.Selector, nvidia.TargetProfile)
	}
	if nvidia.RuntimeBase.Ref != "ghcr.io/kaito-project/aikit/base@"+fixtureChiseledAMD64 {
		t.Errorf("NVIDIA runtime base ref = %q", nvidia.RuntimeBase.Ref)
	}
	if nvidia.RunnerRuntimeBase == nil || nvidia.RunnerRuntimeBase.Ref != "docker.io/library/ubuntu@"+fixtureUbuntu22AMD64 {
		t.Errorf("NVIDIA runner runtime base = %#v", nvidia.RunnerRuntimeBase)
	}
	if got, want := strings.Join(nvidia.Environment, ","), strings.Join(cudaEnvironment(minimumCUDA12), ","); got != want {
		t.Errorf("NVIDIA environment = %q, want %q", got, want)
	}
	if len(nvidia.Fallbacks) != 1 || nvidia.Fallbacks[0] != amd64CPU.Backend {
		t.Fatalf("NVIDIA fallbacks = %#v, want %#v", nvidia.Fallbacks, amd64CPU.Backend)
	}
	if strings.Contains(string(firstJSON), "development") || strings.Contains(string(firstJSON), "latest-") {
		t.Fatalf("generated catalog contains development or mutable latest data:\n%s", firstJSON)
	}
}

func TestRuntimeBaseResolutionFixtures(t *testing.T) {
	resolver := newCachedResolver(readSnapshot(t, "testdata/resolutions.json"))
	tests := []struct {
		name      string
		reference string
		platform  Platform
		want      string
	}{
		{
			name:      "chiseled amd64",
			reference: chiseledRuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureAMD64},
			want:      "ghcr.io/kaito-project/aikit/base@" + fixtureChiseledAMD64,
		},
		{
			name:      "Ubuntu 22.04 amd64",
			reference: ubuntu22RuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureAMD64},
			want:      "docker.io/library/ubuntu@" + fixtureUbuntu22AMD64,
		},
		{
			name:      "Ubuntu 24.04 amd64",
			reference: ubuntuRuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureAMD64},
			want:      "docker.io/library/ubuntu@" + fixtureUbuntuAMD64,
		},
		{
			name:      "Ubuntu 24.04 arm64",
			reference: ubuntuRuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureARM64},
			want:      "docker.io/library/ubuntu@" + fixtureUbuntuARM64,
		},
		{
			name:      "ROCm amd64",
			reference: rocmRuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureAMD64},
			want:      "docker.io/rocm/dev-ubuntu-24.04@" + fixtureROCmAMD64,
		},
		{
			name:      fixtureVulkanARM64Name,
			reference: vulkanRuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureARM64},
			want:      "ghcr.io/kaito-project/aikit/applesilicon/base@" + fixtureVulkanARM64,
		},
		{
			name:      "L4T arm64",
			reference: l4tRuntimeBase,
			platform:  Platform{OS: platformLinux, Architecture: architectureARM64},
			want:      "nvcr.io/nvidia/l4t-jetpack@" + fixtureL4TARM64,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := resolvePlatformReference(context.Background(), resolver, test.reference, test.platform, false)
			if err != nil {
				t.Fatalf("resolvePlatformReference() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("resolvePlatformReference() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestGeneratedDefaultsReachOnlyAvailableEntries(t *testing.T) {
	source := readFixture(t, "testdata/index.yaml")
	catalog, err := Generate(context.Background(), source, GenerateOptions{
		Source:          fixturePin(source),
		Version:         LocalAIVersion,
		CoreRefTemplate: fixtureCoreRefTemplate,
		Resolver:        readSnapshot(t, "testdata/resolutions.json"),
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	tests := []struct {
		name    string
		mutate  func([]Entry) []Entry
		wantErr string
	}{
		{
			name: "generic defaults and platform override resolve",
			mutate: func(entries []Entry) []Entry {
				return entries
			},
		},
		{
			name: "missing platform override target",
			mutate: func(entries []Entry) []Entry {
				return deleteGeneratedEntry(entries, runnerLlamaCpp, selectorNVIDIAL4T, Platform{OS: platformLinux, Architecture: architectureARM64})
			},
			wantErr: "has no entry",
		},
		{
			name: "generic default covers platform exposed by another family",
			mutate: func(entries []Entry) []Entry {
				platform := Platform{OS: platformLinux, Architecture: architectureARM64}
				for _, entry := range entries {
					if entry.Family == runnerLlamaCpp && entry.Selector == selectorDefault && entry.Platform == platform {
						otherFamily := entry
						otherFamily.Family = fixtureFamilyDemo
						entries = append(entries, otherFamily)
						break
					}
				}
				return deleteGeneratedEntry(entries, runnerLlamaCpp, selectorDefault, platform)
			},
			wantErr: "has no entry",
		},
		{
			name: "runtime mismatch",
			mutate: func(entries []Entry) []Entry {
				for index := range entries {
					if entries[index].Family == runnerLlamaCpp && entries[index].Selector == selectorNVIDIAL4T {
						entries[index].Runtime = runtimeCPU
					}
				}
				return entries
			},
			wantErr: "has runtime",
		},
		{
			name: "quarantined default",
			mutate: func(entries []Entry) []Entry {
				for index := range entries {
					if entries[index].Family == runnerLlamaCpp && entries[index].Selector == selectorAMD {
						entries[index].Status = statusQuarantined
					}
				}
				return entries
			},
			wantErr: "unavailable status",
		},
		{
			name: "missing runtime inventory",
			mutate: func(entries []Entry) []Entry {
				filtered := entries[:0]
				for _, entry := range entries {
					if entry.Runtime != runtimeROCm {
						filtered = append(filtered, entry)
					}
				}
				return filtered
			},
			wantErr: "has no entries for runtime",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			entries := append([]Entry(nil), catalog.Entries...)
			err := validateGeneratedDefaultReachability(catalog.Defaults, test.mutate(entries))
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validateGeneratedDefaultReachability() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateGeneratedDefaultReachability() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func deleteGeneratedEntry(entries []Entry, family, selector string, platform Platform) []Entry {
	filtered := entries[:0]
	for _, entry := range entries {
		if entry.Family != family || entry.Selector != selector || entry.Platform != platform {
			filtered = append(filtered, entry)
		}
	}

	return filtered
}

func TestParseSourceMergeKeysAndDuplicateRules(t *testing.T) {
	source := readFixture(t, "testdata/index.yaml")
	entries, err := parseSource(source, fixturePin(source))
	if err != nil {
		t.Fatalf("parseSource() error = %v", err)
	}
	if len(entries) != 9 {
		t.Fatalf("parseSource() entry count = %d, want 9 after exact duplicate collapse", len(entries))
	}
	var concrete sourceEntry
	for _, entry := range entries {
		if entry.Name == fixtureCPULlamaCpp {
			concrete = entry
			break
		}
	}
	if concrete.URI == "" || concrete.Alias != runnerLlamaCpp || concrete.Capabilities[selectorNVIDIA] != "cuda12-llama-cpp" {
		t.Fatalf("merge-expanded concrete entry = %#v", concrete)
	}

	conflicting := []byte("- name: duplicate\n  uri: registry.example/repo:latest-one\n- name: duplicate\n  uri: registry.example/repo:latest-two\n")
	if _, err := parseSource(conflicting, fixturePin(conflicting)); err == nil || !strings.Contains(err.Error(), "conflicting entries") {
		t.Fatalf("parseSource() conflict error = %v", err)
	}
}

func TestVerifySourceRejectsWrongPinAndUnknownYAMLFields(t *testing.T) {
	source := []byte("- name: valid\n")
	pin := fixturePin(source)
	pin.SHA256 = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	if _, err := parseSource(source, pin); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("parseSource() digest error = %v", err)
	}

	unknown := []byte("- name: valid\n  executable: surprise\n")
	if _, err := parseSource(unknown, fixturePin(unknown)); err == nil || !strings.Contains(err.Error(), "unknown field \"executable\"") {
		t.Fatalf("parseSource() unknown field error = %v", err)
	}

	duplicateCapability := []byte("- name: duplicate-capability\n  capabilities:\n    default: cpu-one\n    default: cpu-two\n")
	if _, err := parseSource(duplicateCapability, fixturePin(duplicateCapability)); err == nil || !strings.Contains(err.Error(), "duplicate capability selector") {
		t.Fatalf("parseSource() duplicate capability error = %v", err)
	}

	badRevisionPin := fixturePin(source)
	badRevisionPin.Revision = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
	if _, err := parseSource(source, badRevisionPin); err == nil || !strings.Contains(err.Error(), "not a full Git commit") {
		t.Fatalf("parseSource() revision error = %v", err)
	}
}

func TestPolicyInferencePreservesAcceleratorSemantics(t *testing.T) {
	tests := []struct {
		name            string
		selector        string
		target          string
		platform        Platform
		wantRuntime     string
		wantTarget      string
		wantRuntimeBase string
		wantEnvironment []string
	}{
		{
			name:            "Intel",
			selector:        targetIntel,
			target:          "intel-sycl-f16-demo",
			platform:        Platform{OS: platformLinux, Architecture: architectureAMD64},
			wantRuntime:     runtimeCPU,
			wantTarget:      targetIntel,
			wantRuntimeBase: ubuntuRuntimeBase,
		},
		{
			name:            "Metal",
			selector:        "metal-darwin-arm64",
			target:          "metal-demo",
			platform:        Platform{OS: fixtureOSDarwin, Architecture: architectureARM64},
			wantRuntime:     runtimeApple,
			wantTarget:      targetMetal,
			wantRuntimeBase: vulkanRuntimeBase,
			wantEnvironment: []string{vulkanEnvironment},
		},
		{
			name:            "Vulkan",
			selector:        targetVulkan,
			target:          fixtureVulkanTarget,
			platform:        Platform{OS: platformLinux, Architecture: architectureAMD64},
			wantRuntime:     runtimeCPU,
			wantTarget:      targetVulkan,
			wantRuntimeBase: ubuntuRuntimeBase,
		},
		{
			name:            "L4T CUDA 12",
			selector:        selectorNVIDIAL4T,
			target:          "nvidia-l4t-arm64-demo",
			platform:        Platform{OS: platformLinux, Architecture: architectureARM64},
			wantRuntime:     runtimeCUDA,
			wantTarget:      targetL4TCUDA12,
			wantRuntimeBase: l4tRuntimeBase,
			wantEnvironment: l4tEnvironment(minimumCUDA12),
		},
		{
			name:            "L4T CUDA 13",
			selector:        selectorL4TCUDA13,
			target:          "cuda13-nvidia-l4t-arm64-demo",
			platform:        Platform{OS: platformLinux, Architecture: architectureARM64},
			wantRuntime:     runtimeCUDA,
			wantTarget:      targetL4TCUDA13,
			wantRuntimeBase: ubuntuRuntimeBase,
			wantEnvironment: l4tEnvironment(minimumCUDA13),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			policy, err := policyFor(fixtureFamilyDemo, test.selector, test.target, "", test.platform)
			if err != nil {
				t.Fatalf("policyFor() error = %v", err)
			}
			if policy.Runtime != test.wantRuntime || policy.TargetProfile != test.wantTarget || policy.RuntimeBaseRef != test.wantRuntimeBase ||
				strings.Join(policy.Environment, "\x00") != strings.Join(test.wantEnvironment, "\x00") {
				t.Fatalf("policyFor() = %#v, want runtime %q target %q runtime base %q environment %v", policy, test.wantRuntime, test.wantTarget, test.wantRuntimeBase, test.wantEnvironment)
			}
		})
	}
}

func TestStableVersionReference(t *testing.T) {
	got, err := stableVersionReference("quay.io/example/backend:latest-gpu-demo", "v4.8.2")
	if err != nil {
		t.Fatalf("stableVersionReference() error = %v", err)
	}
	if want := "quay.io/example/backend:v4.8.2-gpu-demo"; got != want {
		t.Fatalf("stableVersionReference() = %q, want %q", got, want)
	}
	if _, err := stableVersionReference("quay.io/example/backend:master-gpu-demo", "v4.8.2"); err == nil {
		t.Fatal("stableVersionReference() accepted a development tag")
	}
	if _, err := stableVersionReference("quay.io/example/backend", "v4.8.2"); err == nil {
		t.Fatal("stableVersionReference() accepted a reference without a tag")
	}
}

func TestGenerateRejectsIncompleteStableMappings(t *testing.T) {
	tests := []struct {
		name    string
		source  string
		wantErr string
	}{
		{
			name:    "missing concrete target",
			source:  "- name: demo\n  capabilities:\n    default: cpu-missing\n",
			wantErr: "targets missing concrete entry",
		},
		{
			name:    "development selectors are not stable inventory",
			source:  "- name: demo-development\n  capabilities:\n    default: cpu-demo-development\n- name: cpu-demo-development\n  uri: registry.example/repo:master-cpu-demo\n",
			wantErr: "no stable selectable entries",
		},
		{
			name:    "stable target must use latest tag",
			source:  "- name: demo\n  capabilities:\n    default: cpu-demo\n- name: cpu-demo\n  uri: registry.example/repo:master-cpu-demo\n",
			wantErr: "does not use a latest tag",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := []byte(test.source)
			_, err := Generate(context.Background(), source, GenerateOptions{
				Source:          fixturePin(source),
				Version:         LocalAIVersion,
				CoreRefTemplate: fixtureCoreRefTemplate,
				Resolver:        failingResolver{},
			})
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("Generate() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func TestGenerateAcceptsMissingWorkloadsAndPlatformlessSpecializedCore(t *testing.T) {
	source := sourceWithDefaultFixture(t, "- name: demo\n  capabilities:\n    default: cpu-demo\n- name: cpu-demo\n  uri: registry.example/repo:latest-cpu-demo\n")
	resolver := scriptedResolver{
		base: readSnapshot(t, "testdata/resolutions.json"),
		manifests: staticResolver{
			"registry.example/repo:v4.8.2-cpu-demo": {{
				Digest:   fixtureDigestA,
				Platform: Platform{OS: platformLinux, Architecture: architectureAMD64},
			}},
			"registry.example/core:v4.8.2-amd64": {{Digest: fixtureDigestB}},
		},
	}
	catalog, err := Generate(context.Background(), source, GenerateOptions{
		Source:          fixturePin(source),
		Version:         LocalAIVersion,
		CoreRefTemplate: fixtureCoreRefTemplate,
		Resolver:        resolver,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	demo := findGeneratedEntry(t, catalog, fixtureFamilyDemo, selectorDefault, Platform{OS: platformLinux, Architecture: architectureAMD64})
	if len(demo.Workloads) != 0 {
		t.Fatalf("demo workloads = %v, want empty", demo.Workloads)
	}
	data, err := Marshal(catalog)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if _, err := backendcatalog.Parse(data); err != nil {
		t.Fatalf("platformless specialized core output is invalid: %v", err)
	}
}

func TestGenerateAppliesReviewedUnavailableSourcePolicyStrictly(t *testing.T) {
	const (
		availableRef = "registry.example/repo:v4.8.2-cpu-demo"
		missingRef   = "quay.io/go-skynet/local-ai-backends:v4.8.2-cpu-kokoros"
		coreRef      = "registry.example/core:v4.8.2-amd64"
	)
	source := sourceWithDefaultFixture(t, `- name: demo
  capabilities:
    default: cpu-demo
- name: cpu-demo
  uri: registry.example/repo:latest-cpu-demo
- name: kokoros
  capabilities:
    default: cpu-kokoros
- name: cpu-kokoros
  uri: quay.io/go-skynet/local-ai-backends:latest-cpu-kokoros
`)
	manifest := ResolvedManifest{
		Digest:   fixtureDigestA,
		Platform: Platform{OS: platformLinux, Architecture: architectureAMD64},
	}
	coreManifest := ResolvedManifest{
		Digest: fixtureDigestB,
	}
	runtimeBaseManifest := ResolvedManifest{
		Digest:   fixtureUbuntuAMD64,
		Platform: Platform{OS: platformLinux, Architecture: architectureAMD64},
	}
	options := GenerateOptions{
		Source:          fixturePin(source),
		Version:         LocalAIVersion,
		CoreRefTemplate: fixtureCoreRefTemplate,
	}

	t.Run("expected missing", func(t *testing.T) {
		options.Resolver = scriptedResolver{
			base:      readSnapshot(t, "testdata/resolutions.json"),
			manifests: staticResolver{availableRef: {manifest}, coreRef: {coreManifest}, ubuntuRuntimeBase: {runtimeBaseManifest}},
			errors: map[string]error{
				missingRef: &ResolutionError{Reference: missingRef, Class: resolutionErrorNotFound, Err: os.ErrNotExist},
			},
		}
		catalog, err := Generate(context.Background(), source, options)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		findGeneratedEntry(t, catalog, fixtureFamilyDemo, selectorDefault, Platform{OS: platformLinux, Architecture: architectureAMD64})
	})

	t.Run("stale policy resolved", func(t *testing.T) {
		options.Resolver = scriptedResolver{base: readSnapshot(t, "testdata/resolutions.json"), manifests: staticResolver{
			availableRef:      {manifest},
			missingRef:        {manifest},
			coreRef:           {coreManifest},
			ubuntuRuntimeBase: {runtimeBaseManifest},
		}}
		_, err := Generate(context.Background(), source, options)
		if err == nil || !strings.Contains(err.Error(), "resolved successfully; remove stale exclusion policy") {
			t.Fatalf("Generate() stale policy error = %v", err)
		}
	})

	t.Run("unexpected error class", func(t *testing.T) {
		options.Resolver = scriptedResolver{
			base:      readSnapshot(t, "testdata/resolutions.json"),
			manifests: staticResolver{availableRef: {manifest}, coreRef: {coreManifest}, ubuntuRuntimeBase: {runtimeBaseManifest}},
			errors:    map[string]error{missingRef: os.ErrPermission},
		}
		_, err := Generate(context.Background(), source, options)
		if err == nil || !strings.Contains(err.Error(), "failed with unexpected class") {
			t.Fatalf("Generate() unexpected-class error = %v", err)
		}
	})
}

func TestUnavailableSourcePolicyCannotOverlapSupportedOverlay(t *testing.T) {
	err := validateUnavailableSourcePolicies([]unavailableSourcePolicy{{
		Version:    LocalAIVersion,
		Family:     runnerLlamaCpp,
		Selector:   selectorDefault,
		SourceRef:  "registry.example/repo:v4.8.2-cpu-llama-cpp",
		ErrorClass: resolutionErrorNotFound,
	}})
	if err == nil || !strings.Contains(err.Error(), "overlaps a supported policy tuple") {
		t.Fatalf("validateUnavailableSourcePolicies() error = %v", err)
	}
}

func TestGenerateExcludesNonLinuxManifests(t *testing.T) {
	source := sourceWithDefaultFixture(t, `- name: demo
  capabilities:
    default: cpu-demo
- name: cpu-demo
  uri: registry.example/repo:latest-cpu-demo
- name: metal-demo
  capabilities:
    metal: metal-demo-target
- name: metal-demo-target
  uri: registry.example/repo:latest-metal-darwin-arm64-demo
`)
	resolver := scriptedResolver{
		base: readSnapshot(t, "testdata/resolutions.json"),
		manifests: staticResolver{
			"registry.example/repo:v4.8.2-cpu-demo": {{
				Digest:   fixtureDigestA,
				Platform: Platform{OS: platformLinux, Architecture: architectureAMD64},
			}},
			"registry.example/repo:v4.8.2-metal-darwin-arm64-demo": {{
				Digest:   fixtureDigestB,
				Platform: Platform{OS: "darwin", Architecture: architectureARM64},
			}},
		},
	}
	catalog, err := Generate(context.Background(), source, GenerateOptions{
		Source:          fixturePin(source),
		Version:         LocalAIVersion,
		CoreRefTemplate: fixtureCoreRefTemplate,
		Resolver:        resolver,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	findGeneratedEntry(t, catalog, fixtureFamilyDemo, selectorDefault, Platform{OS: platformLinux, Architecture: architectureAMD64})
	for _, entry := range catalog.Entries {
		if entry.Platform.OS != platformLinux {
			t.Fatalf("Generate() included non-Linux entry %#v", entry)
		}
	}
}

func TestEntryEligibilityMatchesAIKitRuntimePlatforms(t *testing.T) {
	tests := []struct {
		name          string
		platform      Platform
		runtime       string
		targetProfile string
		want          bool
	}{
		{name: "CPU amd64", platform: Platform{OS: platformLinux, Architecture: architectureAMD64}, runtime: runtimeCPU, targetProfile: runtimeCPU, want: true},
		{name: "CPU arm64", platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeCPU, targetProfile: runtimeCPU, want: true},
		{name: "non-Linux", platform: Platform{OS: fixtureOSDarwin, Architecture: architectureARM64}, runtime: runtimeApple, targetProfile: targetMetal},
		{name: "unsupported architecture", platform: Platform{OS: platformLinux, Architecture: "ppc64le"}, runtime: runtimeCPU, targetProfile: runtimeCPU},
		{name: "Vulkan amd64", platform: Platform{OS: platformLinux, Architecture: architectureAMD64}, runtime: runtimeCPU, targetProfile: targetVulkan, want: true},
		{name: fixtureVulkanARM64Name, platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeApple, targetProfile: targetVulkan, want: true},
		{name: "Vulkan amd64 mislabeled Apple Silicon", platform: Platform{OS: platformLinux, Architecture: architectureAMD64}, runtime: runtimeApple, targetProfile: targetVulkan},
		{name: "Vulkan arm64 mislabeled CPU", platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeCPU, targetProfile: targetVulkan},
		{name: "Metal mislabeled CPU", platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeCPU, targetProfile: targetMetal},
		{name: "ROCm amd64", platform: Platform{OS: platformLinux, Architecture: architectureAMD64}, runtime: runtimeROCm, targetProfile: targetROCm, want: true},
		{name: "ROCm arm64", platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeROCm, targetProfile: targetROCm},
		{name: "L4T amd64", platform: Platform{OS: platformLinux, Architecture: architectureAMD64}, runtime: runtimeCUDA, targetProfile: targetL4TCUDA12},
		{name: "L4T arm64", platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeCUDA, targetProfile: targetL4TCUDA13, want: true},
		{name: "generic CUDA arm64", platform: Platform{OS: platformLinux, Architecture: architectureARM64}, runtime: runtimeCUDA, targetProfile: targetCUDA12, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := entryEligibleForAIKit(test.platform, test.runtime, test.targetProfile); got != test.want {
				t.Fatalf("entryEligibleForAIKit() = %t, want %t", got, test.want)
			}
		})
	}
}

type staticResolver map[string][]ResolvedManifest

func (resolver staticResolver) Resolve(_ context.Context, reference string) ([]ResolvedManifest, error) {
	manifests, ok := resolver[reference]
	if !ok {
		return nil, os.ErrNotExist
	}

	return append([]ResolvedManifest(nil), manifests...), nil
}

type scriptedResolver struct {
	base      Resolver
	manifests staticResolver
	errors    map[string]error
}

func (resolver scriptedResolver) Resolve(ctx context.Context, reference string) ([]ResolvedManifest, error) {
	if err, exists := resolver.errors[reference]; exists {
		return nil, err
	}
	if manifests, exists := resolver.manifests[reference]; exists {
		return append([]ResolvedManifest(nil), manifests...), nil
	}
	if resolver.base != nil {
		return resolver.base.Resolve(ctx, reference)
	}

	return resolver.manifests.Resolve(ctx, reference)
}

type failingResolver struct{}

func (failingResolver) Resolve(_ context.Context, _ string) ([]ResolvedManifest, error) {
	return nil, os.ErrInvalid
}

func readFixture(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %q: %v", path, err)
	}

	return data
}

func readSnapshot(t *testing.T, path string) Resolver {
	t.Helper()
	resolver, err := ParseSnapshot(readFixture(t, path))
	if err != nil {
		t.Fatalf("ParseSnapshot() error = %v", err)
	}

	return resolver
}

func sourceWithDefaultFixture(t *testing.T, extra string) []byte {
	t.Helper()
	source := append([]byte(nil), readFixture(t, "testdata/index.yaml")...)
	if len(source) > 0 && source[len(source)-1] != '\n' {
		source = append(source, '\n')
	}

	return append(source, []byte(extra)...)
}

func findGeneratedEntry(t *testing.T, catalog Catalog, family, selector string, platform Platform) Entry {
	t.Helper()
	for _, entry := range catalog.Entries {
		if entry.Family == family && entry.Selector == selector && entry.Platform == platform {
			return entry
		}
	}
	t.Fatalf("entry %s/%s/%s is missing", family, selector, platform.key())

	return Entry{}
}

func fixturePin(source []byte) SourcePin {
	digest := sha256.Sum256(source)

	return SourcePin{
		Repository: "https://example.com/localai",
		Path:       "backend/index.yaml",
		Revision:   "1111111111111111111111111111111111111111",
		SHA256:     "sha256:" + hex.EncodeToString(digest[:]),
	}
}
