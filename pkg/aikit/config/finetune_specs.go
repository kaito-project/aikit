package config

const (
	defaultMaxSeqLength              = 2048
	defaultLoadIn4bit                = true
	defaultBatchSize                 = 2
	defaultGradientAccumulationSteps = 4
	defaultWarmupSteps               = 10
	defaultMaxSteps                  = 60
	defaultLearningRate              = 0.0002
	defaultLoggingSteps              = 1
	defaultOptimizer                 = "adamw_8bit"
	defaultWeightDecay               = 0.01
	defaultLrSchedulerType           = "linear"
	defaultSeed                      = 42
	defaultOutputQuantize            = "q4_k_m"
	defaultOutputName                = "aikit-model"
)

type FineTuneConfig struct {
	APIVersion string             `yaml:"apiVersion"`
	Target     string             `yaml:"target"`
	BaseModel  string             `yaml:"baseModel"`
	Datasets   []Dataset          `yaml:"datasets"`
	Config     FineTuneConfigSpec `yaml:"config"`
	Output     FineTuneOutputSpec `yaml:"output"`
}

// UnmarshalYAML normalizes fine-tuning defaults while preserving explicitly configured zero values.
func (c *FineTuneConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var raw rawFineTuneConfig
	if err := unmarshal(&raw); err != nil {
		return err
	}

	*c = raw.normalize()
	return nil
}

type FineTuneConfigSpec struct {
	Unsloth FineTuneConfigUnslothSpec `yaml:"unsloth"`
}

type Dataset struct {
	Source string `yaml:"source"`
	Type   string `yaml:"type"`
}

type FineTuneConfigUnslothSpec struct {
	Packing                   bool    `yaml:"packing"`
	MaxSeqLength              int     `yaml:"maxSeqLength"`
	LoadIn4bit                bool    `yaml:"loadIn4bit"`
	BatchSize                 int     `yaml:"batchSize"`
	GradientAccumulationSteps int     `yaml:"gradientAccumulationSteps"`
	WarmupSteps               int     `yaml:"warmupSteps"`
	MaxSteps                  int     `yaml:"maxSteps"`
	LearningRate              float64 `yaml:"learningRate"`
	LoggingSteps              int     `yaml:"loggingSteps"`
	Optimizer                 string  `yaml:"optimizer"`
	WeightDecay               float64 `yaml:"weightDecay"`
	LrSchedulerType           string  `yaml:"lrSchedulerType"`
	Seed                      int     `yaml:"seed"`
}

type FineTuneOutputSpec struct {
	Quantize string `yaml:"quantize"`
	Name     string `yaml:"name"`
}

type rawFineTuneConfig struct {
	APIVersion string                 `yaml:"apiVersion"`
	Target     string                 `yaml:"target"`
	BaseModel  string                 `yaml:"baseModel"`
	Datasets   []Dataset              `yaml:"datasets"`
	Config     *rawFineTuneConfigSpec `yaml:"config"`
	Output     *rawFineTuneOutputSpec `yaml:"output"`
}

type rawFineTuneConfigSpec struct {
	Unsloth *rawFineTuneConfigUnslothSpec `yaml:"unsloth"`
}

type rawFineTuneConfigUnslothSpec struct {
	Packing                   *bool    `yaml:"packing"`
	MaxSeqLength              *int     `yaml:"maxSeqLength"`
	LoadIn4bit                *bool    `yaml:"loadIn4bit"`
	BatchSize                 *int     `yaml:"batchSize"`
	GradientAccumulationSteps *int     `yaml:"gradientAccumulationSteps"`
	WarmupSteps               *int     `yaml:"warmupSteps"`
	MaxSteps                  *int     `yaml:"maxSteps"`
	LearningRate              *float64 `yaml:"learningRate"`
	LoggingSteps              *int     `yaml:"loggingSteps"`
	Optimizer                 *string  `yaml:"optimizer"`
	WeightDecay               *float64 `yaml:"weightDecay"`
	LrSchedulerType           *string  `yaml:"lrSchedulerType"`
	Seed                      *int     `yaml:"seed"`
}

type rawFineTuneOutputSpec struct {
	Quantize *string `yaml:"quantize"`
	Name     *string `yaml:"name"`
}

func (c rawFineTuneConfig) normalize() FineTuneConfig {
	var rawUnsloth *rawFineTuneConfigUnslothSpec
	if c.Config != nil {
		rawUnsloth = c.Config.Unsloth
	}

	return FineTuneConfig{
		APIVersion: c.APIVersion,
		Target:     c.Target,
		BaseModel:  c.BaseModel,
		Datasets:   c.Datasets,
		Config: FineTuneConfigSpec{
			Unsloth: normalizeUnslothConfig(rawUnsloth),
		},
		Output: normalizeFineTuneOutput(c.Output),
	}
}

func normalizeUnslothConfig(c *rawFineTuneConfigUnslothSpec) FineTuneConfigUnslothSpec {
	if c == nil {
		c = &rawFineTuneConfigUnslothSpec{}
	}

	return FineTuneConfigUnslothSpec{
		Packing:                   valueOrDefault(c.Packing, false),
		MaxSeqLength:              valueOrDefault(c.MaxSeqLength, defaultMaxSeqLength),
		LoadIn4bit:                valueOrDefault(c.LoadIn4bit, defaultLoadIn4bit),
		BatchSize:                 valueOrDefault(c.BatchSize, defaultBatchSize),
		GradientAccumulationSteps: valueOrDefault(c.GradientAccumulationSteps, defaultGradientAccumulationSteps),
		WarmupSteps:               valueOrDefault(c.WarmupSteps, defaultWarmupSteps),
		MaxSteps:                  valueOrDefault(c.MaxSteps, defaultMaxSteps),
		LearningRate:              valueOrDefault(c.LearningRate, defaultLearningRate),
		LoggingSteps:              valueOrDefault(c.LoggingSteps, defaultLoggingSteps),
		Optimizer:                 valueOrDefault(c.Optimizer, defaultOptimizer),
		WeightDecay:               valueOrDefault(c.WeightDecay, defaultWeightDecay),
		LrSchedulerType:           valueOrDefault(c.LrSchedulerType, defaultLrSchedulerType),
		Seed:                      valueOrDefault(c.Seed, defaultSeed),
	}
}

func normalizeFineTuneOutput(c *rawFineTuneOutputSpec) FineTuneOutputSpec {
	if c == nil {
		c = &rawFineTuneOutputSpec{}
	}

	return FineTuneOutputSpec{
		Quantize: valueOrDefault(c.Quantize, defaultOutputQuantize),
		Name:     valueOrDefault(c.Name, defaultOutputName),
	}
}

func valueOrDefault[T any](value *T, fallback T) T {
	if value == nil {
		return fallback
	}
	return *value
}
