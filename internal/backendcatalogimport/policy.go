package backendcatalogimport

import (
	"fmt"
	"sort"
	"strings"
)

type fallbackTarget struct {
	Family   string
	Selector string
}

type entryPolicy struct {
	Runtime              string
	TargetProfile        string
	Status               string
	RuntimeBaseRef       string
	RunnerRuntimeBaseRef string
	InstallName          string
	SystemPackages       []string
	RuntimeSymlinks      []RuntimeSymlink
	Environment          []string
	RunnerProfile        string
	Fallbacks            []fallbackTarget
}

var reviewedSystemPackages = map[string][]string{
	familyVLLM: {systemPackageGCC, systemPackageLibcDev},
}

var rocmRuntimeSymlinks = []RuntimeSymlink{{
	Target: "libhipblaslt.so.1",
	Path:   "/opt/rocm/lib/libhipblaslt.so.0",
}}

func hasSupportedOverlay(family, selector string) bool {
	switch family + "/" + selector {
	case runnerLlamaCpp + "/" + selectorDefault, runnerLlamaCpp + "/" + selectorNVIDIA,
		familyDiffusers + "/" + selectorNVIDIA, familyVLLM + "/" + selectorNVIDIA,
		familyVLLMCpp + "/" + selectorDefault, familyVLLMCpp + "/" + selectorNVIDIA:
		return true
	default:
		return false
	}
}

func artifactVersionFor(requestedVersion, family, selector string) string {
	if requestedVersion != LocalAIVersion {
		return requestedVersion
	}

	switch family + "/" + selector {
	case familyDiffusers + "/" + selectorNVIDIA, runnerLlamaCpp + "/" + targetVulkan:
		return legacyLocalAIVersion
	default:
		return requestedVersion
	}
}

func coreReferenceTemplateForVersion(template, requestedVersion, entryVersion string) (string, error) {
	if requestedVersion == entryVersion {
		return template, nil
	}
	if strings.Count(template, requestedVersion) != 1 {
		return "", fmt.Errorf("core reference template %q must contain imported version %q exactly once for compatibility pin %q", template, requestedVersion, entryVersion)
	}

	return strings.Replace(template, requestedVersion, entryVersion, 1), nil
}

func policyFor(family, selector, target, sourceRef string, platform Platform) (entryPolicy, error) {
	targetProfile, err := inferTargetProfile(selector, target, sourceRef)
	if err != nil {
		return entryPolicy{}, err
	}

	policy := entryPolicy{
		Runtime:        inferRuntime(targetProfile, platform),
		TargetProfile:  targetProfile,
		Status:         statusExperimental,
		RuntimeBaseRef: runtimeBaseReference(targetProfile, platform),
		SystemPackages: append([]string(nil), reviewedSystemPackages[family]...),
		Environment:    runtimeEnvironment(targetProfile, target, platform),
		RunnerProfile:  runnerUnsupported,
	}
	if targetProfile == targetROCm {
		policy.SystemPackages = append(policy.SystemPackages, "pciutils")
		policy.RuntimeSymlinks = append([]RuntimeSymlink(nil), rocmRuntimeSymlinks...)
	}
	if family == familyVLLM && targetProfile == targetCUDA12 && platform.OS == platformLinux && platform.Architecture == architectureAMD64 {
		policy.Environment = append(policy.Environment, vllmNativeSampler)
	}
	sort.Strings(policy.SystemPackages)
	sort.Strings(policy.Environment)

	key := family + "/" + selector + "/" + platform.OS + "/" + platform.Architecture
	switch key {
	case runnerLlamaCpp + "/" + selectorDefault + "/linux/amd64", runnerLlamaCpp + "/" + selectorDefault + "/linux/arm64":
		policy.Status = statusSupported
		policy.RuntimeBaseRef = chiseledRuntimeBase
		policy.RunnerRuntimeBaseRef = ubuntu22RuntimeBase
		policy.RunnerProfile = runnerLlamaCpp
	case runnerLlamaCpp + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.RuntimeBaseRef = chiseledRuntimeBase
		policy.RunnerRuntimeBaseRef = ubuntu22RuntimeBase
		policy.RunnerProfile = runnerLlamaCpp
		policy.Fallbacks = []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}}
	case runnerLlamaCpp + "/" + selectorAMD + "/linux/amd64":
		policy.InstallName = "hipblas-llama-cpp"
		policy.RunnerProfile = runnerLlamaCpp
		policy.Fallbacks = []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}}
	case runnerLlamaCpp + "/" + selectorNVIDIAL4T + "/linux/arm64",
		runnerLlamaCpp + "/" + selectorL4TCUDA12 + "/linux/arm64",
		runnerLlamaCpp + "/" + selectorL4TCUDA13 + "/linux/arm64":
		policy.RunnerProfile = runnerLlamaCpp
		policy.Fallbacks = []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}}
	case runnerLlamaCpp + "/" + targetVulkan + "/linux/arm64":
		policy.InstallName = "gpu-vulkan-llama-cpp"
	case familyDiffusers + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.RuntimeBaseRef = ubuntu22RuntimeBase
		policy.RunnerProfile = runnerHFConfig
	case familyVLLM + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.RuntimeBaseRef = ubuntu22RuntimeBase
		policy.RunnerProfile = runnerHFConfig
	case familyVLLMCpp + "/" + selectorDefault + "/linux/amd64", familyVLLMCpp + "/" + selectorDefault + "/linux/arm64":
		policy.Status = statusSupported
		policy.RuntimeBaseRef = chiseledRuntimeBase
		policy.RunnerRuntimeBaseRef = ubuntu22RuntimeBase
		policy.RunnerProfile = familyVLLMCpp
	case familyVLLMCpp + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.RuntimeBaseRef = chiseledRuntimeBase
		policy.RunnerRuntimeBaseRef = ubuntu22RuntimeBase
		policy.RunnerProfile = familyVLLMCpp
	}

	// Kokoro ignores baked model paths and downloads an unpinned model at
	// runtime. Keep every tuple unavailable until the backend can consume the
	// immutable model materialized by AIKit.
	if family == familyKokoro {
		policy.Status = statusQuarantined
	}

	// LocalAI v4.8.2's CUDA 12 SGLang bundle mixes PyTorch CUDA 13.0 with
	// TorchAudio CUDA 12.8 and exits before its gRPC service becomes ready.
	if family == familySGLang && policy.TargetProfile == targetCUDA12 && platform.OS == platformLinux && platform.Architecture == architectureAMD64 {
		policy.Status = statusQuarantined
	}

	return policy, nil
}

func runtimeBaseReference(targetProfile string, platform Platform) string {
	switch targetProfile {
	case targetROCm:
		return rocmRuntimeBase
	case targetMetal:
		return vulkanRuntimeBase
	case targetVulkan:
		if platform.Architecture == architectureARM64 {
			return vulkanRuntimeBase
		}
		return ubuntuRuntimeBase
	case targetL4TCUDA12:
		return l4tRuntimeBase
	case targetL4TCUDA13:
		return ubuntuRuntimeBase
	default:
		return ubuntuRuntimeBase
	}
}

func runtimeEnvironment(targetProfile, target string, platform Platform) []string {
	switch targetProfile {
	case targetCUDA12:
		if strings.HasPrefix(target, "vulkan-") {
			return cudaEnvironmentWithCapabilities(minimumCUDA12, cudaAllCapabilities)
		}
		return cudaEnvironment(minimumCUDA12)
	case targetCUDA13:
		if strings.HasPrefix(target, "vulkan-") {
			return cudaEnvironmentWithCapabilities(minimumCUDA13, cudaAllCapabilities)
		}
		return cudaEnvironment(minimumCUDA13)
	case targetL4TCUDA12:
		return l4tEnvironment(minimumCUDA12)
	case targetL4TCUDA13:
		return l4tEnvironment(minimumCUDA13)
	case targetROCm:
		return []string{
			"LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib",
			"LOCALAI_FORCE_META_BACKEND_CAPABILITY=amd",
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin",
		}
	case targetMetal:
		return []string{vulkanEnvironment}
	case targetVulkan:
		if platform.Architecture == architectureARM64 {
			return []string{vulkanEnvironment}
		}
		return []string{}
	default:
		return []string{}
	}
}

func cudaEnvironment(minimumVersion string) []string {
	return cudaEnvironmentWithCapabilities(minimumVersion, cudaCapabilities)
}

func cudaEnvironmentWithCapabilities(minimumVersion, capabilities string) []string {
	return []string{
		cudaBuildType,
		capabilities,
		"NVIDIA_REQUIRE_CUDA=cuda>=" + minimumVersion,
		cudaVisibleDevices,
	}
}

func l4tEnvironment(minimumVersion string) []string {
	return []string{
		cudaBuildType,
		cudaHome,
		cudaLibraryPath,
		cudaAllCapabilities,
		"NVIDIA_REQUIRE_CUDA=cuda>=" + minimumVersion,
		cudaVisibleDevices,
		cudaPath,
	}
}

func inferTargetProfile(selector, target, sourceRef string) (string, error) {
	switch {
	case strings.HasPrefix(target, "cuda13-nvidia-l4t-"), selector == selectorL4TCUDA13,
		strings.Contains(sourceRef, "-nvidia-l4t-cuda-13-"):
		return targetL4TCUDA13, nil
	case strings.HasPrefix(target, "nvidia-l4t-"), selector == selectorNVIDIAL4T, selector == selectorL4TCUDA12:
		return targetL4TCUDA12, nil
	case strings.HasPrefix(target, "cuda13-"), selector == selectorNVIDIACUDA13:
		return targetCUDA13, nil
	case strings.HasPrefix(target, "cuda12-"), selector == selectorNVIDIA, selector == selectorNVIDIACUDA12:
		return targetCUDA12, nil
	case strings.HasPrefix(target, "rocm-"), selector == selectorAMD:
		return targetROCm, nil
	case strings.HasPrefix(target, "metal-"), strings.HasPrefix(selector, "metal"):
		return targetMetal, nil
	case strings.HasPrefix(target, "vulkan-"), selector == "vulkan":
		return targetVulkan, nil
	case strings.HasPrefix(target, "intel-"), selector == targetIntel:
		return targetIntel, nil
	case strings.HasPrefix(target, "cpu-"), selector == selectorDefault, selector == runtimeCPU:
		return runtimeCPU, nil
	default:
		return "", fmt.Errorf("cannot infer target profile for selector %q target %q", selector, target)
	}
}

func inferRuntime(targetProfile string, platform Platform) string {
	switch targetProfile {
	case targetCUDA12, targetCUDA13, targetL4TCUDA12, targetL4TCUDA13:
		return runtimeCUDA
	case targetROCm:
		return runtimeROCm
	case targetMetal:
		return runtimeApple
	case targetVulkan:
		if platform.Architecture == architectureARM64 {
			return runtimeApple
		}
		return runtimeCPU
	default:
		return runtimeCPU
	}
}
