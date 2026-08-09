package backendcatalogimport

import (
	"fmt"
	"strings"
)

type fallbackTarget struct {
	Family   string
	Selector string
}

type entryPolicy struct {
	Runtime           string
	TargetProfile     string
	Status            string
	DependencyProfile string
	RunnerProfile     string
	Base              string
	SelfContained     bool
	MinimumCUDA       string
	Fallbacks         []fallbackTarget
}

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

func policyFor(family, selector, target string, platform Platform) (entryPolicy, error) {
	targetProfile, err := inferTargetProfile(selector, target)
	if err != nil {
		return entryPolicy{}, err
	}

	policy := entryPolicy{
		Runtime:           inferRuntime(targetProfile),
		TargetProfile:     targetProfile,
		Status:            statusExperimental,
		DependencyProfile: dependencyNone,
		RunnerProfile:     runnerUnsupported,
		Base:              baseUbuntu,
	}
	switch targetProfile {
	case targetCUDA12, targetL4TCUDA12:
		policy.MinimumCUDA = minimumCUDA12
	case targetCUDA13, targetL4TCUDA13:
		policy.MinimumCUDA = minimumCUDA13
	case targetROCm:
		policy.Base = baseUbuntu24
	case targetMetal, targetVulkan:
		policy.Base = baseAppleSilicon
	}

	key := family + "/" + selector + "/" + platform.OS + "/" + platform.Architecture
	switch key {
	case runnerLlamaCpp + "/" + selectorDefault + "/linux/amd64", runnerLlamaCpp + "/" + selectorDefault + "/linux/arm64":
		policy.Status = statusSupported
		policy.RunnerProfile = runnerLlamaCpp
		policy.Base = baseDistroless
		policy.SelfContained = true
	case runnerLlamaCpp + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.RunnerProfile = runnerLlamaCpp
		policy.Base = baseDistroless
		policy.SelfContained = true
		policy.MinimumCUDA = minimumCUDA12
		policy.Fallbacks = []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}}
	case runnerLlamaCpp + "/" + selectorAMD + "/linux/amd64":
		policy.Fallbacks = []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}}
	case familyDiffusers + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.DependencyProfile = familyDiffusers
		policy.RunnerProfile = runnerHFConfig
		policy.MinimumCUDA = minimumCUDA12
	case familyVLLM + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.DependencyProfile = familyVLLM
		policy.RunnerProfile = runnerHFConfig
		policy.MinimumCUDA = minimumCUDA12
	case familyVLLMCpp + "/" + selectorDefault + "/linux/amd64", familyVLLMCpp + "/" + selectorDefault + "/linux/arm64":
		policy.Status = statusSupported
		policy.RunnerProfile = familyVLLMCpp
		policy.Base = baseDistroless
		policy.SelfContained = true
	case familyVLLMCpp + "/" + selectorNVIDIA + "/linux/amd64":
		policy.Status = statusSupported
		policy.RunnerProfile = familyVLLMCpp
		policy.Base = baseDistroless
		policy.SelfContained = true
		policy.MinimumCUDA = minimumCUDA13
	}

	return policy, nil
}

func inferTargetProfile(selector, target string) (string, error) {
	switch {
	case strings.HasPrefix(target, "cuda13-nvidia-l4t-"), selector == "nvidia-l4t-cuda-13":
		return targetL4TCUDA13, nil
	case strings.HasPrefix(target, "nvidia-l4t-"), selector == "nvidia-l4t", selector == "nvidia-l4t-cuda-12":
		return targetL4TCUDA12, nil
	case strings.HasPrefix(target, "cuda13-"), selector == "nvidia-cuda-13":
		return targetCUDA13, nil
	case strings.HasPrefix(target, "cuda12-"), selector == selectorNVIDIA, selector == "nvidia-cuda-12":
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

func inferRuntime(targetProfile string) string {
	switch targetProfile {
	case targetCUDA12, targetCUDA13, targetL4TCUDA12, targetL4TCUDA13:
		return runtimeCUDA
	case targetROCm:
		return runtimeROCm
	case targetMetal, targetVulkan:
		return runtimeApple
	default:
		return runtimeCPU
	}
}
