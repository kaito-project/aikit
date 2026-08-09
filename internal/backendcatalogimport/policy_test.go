package backendcatalogimport

import "testing"

func TestReviewedPolicyOverlay(t *testing.T) {
	tests := []struct {
		name          string
		family        string
		selector      string
		target        string
		architecture  string
		status        string
		dependency    string
		runner        string
		base          string
		selfContained bool
		minimumCUDA   string
		fallbacks     int
	}{
		{name: "llama CPU amd64", family: runnerLlamaCpp, selector: selectorDefault, target: fixtureCPULlamaCpp, architecture: architectureAMD64, status: statusSupported, dependency: dependencyNone, runner: runnerLlamaCpp, base: baseDistroless, selfContained: true},
		{name: "llama CPU arm64", family: runnerLlamaCpp, selector: selectorDefault, target: fixtureCPULlamaCpp, architecture: architectureARM64, status: statusSupported, dependency: dependencyNone, runner: runnerLlamaCpp, base: baseDistroless, selfContained: true},
		{name: "llama CUDA", family: runnerLlamaCpp, selector: selectorNVIDIA, target: "cuda12-llama-cpp", architecture: architectureAMD64, status: statusSupported, dependency: dependencyNone, runner: runnerLlamaCpp, base: baseDistroless, selfContained: true, minimumCUDA: minimumCUDA12, fallbacks: 1},
		{name: "llama ROCm", family: runnerLlamaCpp, selector: selectorAMD, target: "rocm-llama-cpp", architecture: architectureAMD64, status: statusExperimental, dependency: dependencyNone, runner: runnerUnsupported, base: baseUbuntu24, fallbacks: 1},
		{name: "llama Vulkan", family: runnerLlamaCpp, selector: targetVulkan, target: "vulkan-llama-cpp", architecture: architectureARM64, status: statusExperimental, dependency: dependencyNone, runner: runnerUnsupported, base: baseAppleSilicon},
		{name: "unreviewed ROCm uses runtime base", family: fixtureFamilyDemo, selector: selectorAMD, target: "rocm-demo", architecture: architectureAMD64, status: statusExperimental, dependency: dependencyNone, runner: runnerUnsupported, base: baseUbuntu24},
		{name: "unreviewed Vulkan uses runtime base", family: fixtureFamilyDemo, selector: targetVulkan, target: "vulkan-demo", architecture: architectureARM64, status: statusExperimental, dependency: dependencyNone, runner: runnerUnsupported, base: baseAppleSilicon},
		{name: "diffusers CUDA", family: familyDiffusers, selector: selectorNVIDIA, target: "cuda12-diffusers", architecture: architectureAMD64, status: statusSupported, dependency: familyDiffusers, runner: runnerHFConfig, base: baseUbuntu, minimumCUDA: minimumCUDA12},
		{name: "vllm CUDA", family: familyVLLM, selector: selectorNVIDIA, target: "cuda12-vllm", architecture: architectureAMD64, status: statusSupported, dependency: familyVLLM, runner: runnerHFConfig, base: baseUbuntu, minimumCUDA: minimumCUDA12},
		{name: "vllm-cpp CPU", family: familyVLLMCpp, selector: selectorDefault, target: "cpu-vllm-cpp", architecture: architectureAMD64, status: statusSupported, dependency: dependencyNone, runner: familyVLLMCpp, base: baseDistroless, selfContained: true},
		{name: "vllm-cpp CUDA", family: familyVLLMCpp, selector: selectorNVIDIA, target: "cuda13-vllm-cpp", architecture: architectureAMD64, status: statusSupported, dependency: dependencyNone, runner: familyVLLMCpp, base: baseDistroless, selfContained: true, minimumCUDA: minimumCUDA13},
		{name: "unreviewed tuple defaults", family: fixtureFamilyDemo, selector: selectorNVIDIA, target: "cuda12-demo", architecture: architectureAMD64, status: statusExperimental, dependency: dependencyNone, runner: runnerUnsupported, base: baseUbuntu, minimumCUDA: minimumCUDA12},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			policy, err := policyFor(test.family, test.selector, test.target, Platform{OS: platformLinux, Architecture: test.architecture})
			if err != nil {
				t.Fatalf("policyFor() error = %v", err)
			}
			if policy.Status != test.status || policy.DependencyProfile != test.dependency || policy.RunnerProfile != test.runner || policy.Base != test.base ||
				policy.SelfContained != test.selfContained || policy.MinimumCUDA != test.minimumCUDA || len(policy.Fallbacks) != test.fallbacks {
				t.Fatalf("policyFor() = %#v", policy)
			}
		})
	}
}
