package config

// InferenceConfig is the root aikitfile schema for building an inference image
// that serves one or more models.
type InferenceConfig struct {
	// APIVersion is the aikitfile schema version. Must be v1alpha1.
	APIVersion string `yaml:"apiVersion"`
	// Debug enables verbose debug logging in the built image.
	Debug bool `yaml:"debug"`
	// Runtime selects the hardware acceleration runtime (e.g. cuda, rocm, applesilicon). Empty means CPU.
	Runtime string `yaml:"runtime"`
	// Backends lists the inference backends to install (e.g. llama-cpp, diffusers, vllm).
	Backends []string `yaml:"backends"`
	// Models lists the models to embed in or download into the image.
	Models []Model `yaml:"models"`
	// Config is the raw LocalAI model configuration appended verbatim to the image.
	Config string `yaml:"config"`
}

// Model describes a single model to make available in the inference image.
type Model struct {
	// Name is the model identifier clients use to select this model.
	Name string `yaml:"name"`
	// Source is the URL or OCI reference the model is downloaded from.
	Source string `yaml:"source"`
	// SHA256 is the optional checksum used to verify the downloaded model.
	SHA256 string `yaml:"sha256"`
	// PromptTemplates are the named chat/prompt templates registered for this model.
	PromptTemplates []PromptTemplate `yaml:"promptTemplates"`
}

// PromptTemplate is a named Go text/template used to format prompts for a model.
type PromptTemplate struct {
	// Name is the template identifier referenced by the model configuration.
	Name string `yaml:"name"`
	// Template is the Go text/template body used to render the prompt.
	Template string `yaml:"template"`
}
