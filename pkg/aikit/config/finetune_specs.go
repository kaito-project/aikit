package config

// FineTuneConfig is the root aikitfile schema for a fine-tuning job that
// produces a customized model from a base model and datasets.
type FineTuneConfig struct {
	// APIVersion is the aikitfile schema version. Must be v1alpha1.
	APIVersion string `yaml:"apiVersion"`
	// Target selects the fine-tuning implementation to use (e.g. unsloth).
	Target string `yaml:"target"`
	// BaseModel is the model the fine-tuning job starts from.
	BaseModel string `yaml:"baseModel"`
	// Datasets lists the training datasets used to fine-tune the base model.
	Datasets []Dataset `yaml:"datasets"`
	// Config holds the target-specific fine-tuning hyperparameters.
	Config FineTuneConfigSpec `yaml:"config"`
	// Output controls the format and name of the produced model.
	Output FineTuneOutputSpec `yaml:"output"`
}

// FineTuneConfigSpec groups the per-target fine-tuning configuration blocks.
type FineTuneConfigSpec struct {
	// Unsloth holds hyperparameters for the unsloth fine-tuning target.
	Unsloth FineTuneConfigUnslothSpec `yaml:"unsloth"`
}

// Dataset describes a single training dataset for a fine-tuning job.
type Dataset struct {
	// Source is the URL or dataset reference the training data is loaded from.
	Source string `yaml:"source"`
	// Type is the dataset format (e.g. alpaca).
	Type string `yaml:"type"`
}

// FineTuneConfigUnslothSpec holds the unsloth fine-tuning hyperparameters.
type FineTuneConfigUnslothSpec struct {
	// Packing enables sequence packing to pack multiple short samples into one sequence.
	Packing bool `yaml:"packing"`
	// MaxSeqLength is the maximum token sequence length for training (default 2048).
	MaxSeqLength int `yaml:"maxSeqLength"`
	// LoadIn4bit enables 4-bit quantized loading of the base model to reduce memory use.
	LoadIn4bit bool `yaml:"loadIn4bit"`
	// BatchSize is the per-device training batch size (default 2).
	BatchSize int `yaml:"batchSize"`
	// GradientAccumulationSteps is the number of steps to accumulate gradients before an update (default 4).
	GradientAccumulationSteps int `yaml:"gradientAccumulationSteps"`
	// WarmupSteps is the number of learning-rate warmup steps (default 10).
	WarmupSteps int `yaml:"warmupSteps"`
	// MaxSteps is the total number of training steps to run (default 60).
	MaxSteps int `yaml:"maxSteps"`
	// LearningRate is the optimizer learning rate (default 0.0002).
	LearningRate float64 `yaml:"learningRate"`
	// LoggingSteps is how often, in steps, training metrics are logged (default 1).
	LoggingSteps int `yaml:"loggingSteps"`
	// Optimizer is the optimizer algorithm to use (default adamw_8bit).
	Optimizer string `yaml:"optimizer"`
	// WeightDecay is the weight-decay regularization factor (default 0.01).
	WeightDecay float64 `yaml:"weightDecay"`
	// LrSchedulerType is the learning-rate scheduler type (default linear).
	LrSchedulerType string `yaml:"lrSchedulerType"`
	// Seed is the random seed used for reproducible training (default 42).
	Seed int `yaml:"seed"`
}

// FineTuneOutputSpec controls how the fine-tuned model is exported.
type FineTuneOutputSpec struct {
	// Quantize is the GGUF quantization format applied to the output model (default q4_k_m).
	Quantize string `yaml:"quantize"`
	// Name is the file name of the produced model (default aikit-model).
	Name string `yaml:"name"`
}
