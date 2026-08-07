package config

import (
	"fmt"
	"strings"

	"github.com/kaito-project/aikit/pkg/utils"
	yaml "gopkg.in/yaml.v2"
)

const (
	defaultMaxSeqLength              = 2048
	defaultLoadIn4bit                = true
	defaultLoss                      = utils.SFTLossAll
	defaultBatchSize                 = 2
	defaultGradientAccumulationSteps = 4
	defaultWarmupSteps               = 10
	defaultMaxSteps                  = 60
	defaultSFTLearningRate           = 0.0002
	defaultDPOLearningRate           = 0.000001
	defaultLoggingSteps              = 1
	defaultOptimizer                 = "adamw_8bit"
	defaultWeightDecay               = 0.01
	defaultLrSchedulerType           = "linear"
	defaultSeed                      = 42
	defaultOutputQuantize            = "q4_k_m"
	defaultOutputName                = "aikit-model"
	defaultDatasetSplit              = "train"
	datasetLoaderFieldType           = "type"
	datasetLoaderFieldSubset         = "subset"
	datasetLoaderFieldSplit          = "split"
	datasetLoaderFieldRevision       = "revision"
	datasetLoaderFieldChecksum       = "checksum"
	defaultDPOBeta                   = 0.1
	defaultDPOLossType               = utils.DPOLossSigmoid
	defaultDPOMaxPromptLength        = 512
)

type FineTuneConfig struct {
	APIVersion string                `yaml:"apiVersion"`
	Target     string                `yaml:"target"`
	BaseModel  string                `yaml:"baseModel"`
	Objective  FineTuneObjectiveSpec `yaml:"objective,omitempty"`
	Datasets   []Dataset             `yaml:"datasets"`
	Config     FineTuneConfigSpec    `yaml:"config"`
	Output     FineTuneOutputSpec    `yaml:"output"`
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

// FineTuneObjectiveSpec selects the training objective independently of the dataset record schema.
type FineTuneObjectiveSpec struct {
	Type            string  `yaml:"type"`
	Beta            float64 `yaml:"beta,omitempty"`
	LossType        string  `yaml:"lossType,omitempty"`
	MaxPromptLength int     `yaml:"maxPromptLength,omitempty"`

	betaConfigured            bool
	lossTypeConfigured        bool
	maxPromptLengthConfigured bool
}

// HasDPOSettings reports whether DPO-only settings were explicitly configured or populated.
func (o FineTuneObjectiveSpec) HasDPOSettings() bool {
	return o.betaConfigured || o.lossTypeConfigured || o.maxPromptLengthConfigured ||
		o.Beta != 0 || o.LossType != "" || o.MaxPromptLength != 0
}

type Dataset struct {
	Source string             `yaml:"source"`
	Type   string             `yaml:"type"`
	Loader *DatasetLoaderSpec `yaml:"loader,omitempty"`
}

// DatasetLoaderSpec separates source loading from the training record schema.
type DatasetLoaderSpec struct {
	Type     string `yaml:"type"`
	Subset   string `yaml:"subset,omitempty"`
	Split    string `yaml:"split"`
	Revision string `yaml:"revision,omitempty"`
	Checksum string `yaml:"checksum,omitempty"`
}

// UnmarshalYAML applies loader defaults while rejecting unknown or non-string nested fields.
func (d *Dataset) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawDataset struct {
		Source string `yaml:"source"`
		Type   string `yaml:"type"`
	}

	var raw rawDataset
	if err := unmarshal(&raw); err != nil {
		return err
	}

	var fields yaml.MapSlice
	if err := unmarshal(&fields); err != nil {
		return err
	}

	*d = Dataset{Source: raw.Source, Type: raw.Type}
	loaderFound := false
	for _, field := range fields {
		fieldName, ok := field.Key.(string)
		if !ok || fieldName != "loader" {
			continue
		}
		if loaderFound {
			return fmt.Errorf("datasets[].loader is defined more than once")
		}
		loaderFound = true

		loaderFields, ok := field.Value.(yaml.MapSlice)
		if !ok {
			return fmt.Errorf("datasets[].loader must be a mapping")
		}
		loader, err := decodeDatasetLoaderSpec(loaderFields)
		if err != nil {
			return err
		}
		d.Loader = loader
	}

	return nil
}

func decodeDatasetLoaderSpec(fields yaml.MapSlice) (*DatasetLoaderSpec, error) {
	loader := &DatasetLoaderSpec{Split: defaultDatasetSplit}
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		fieldName, ok := field.Key.(string)
		if !ok {
			return nil, fmt.Errorf("datasets[].loader field names must be strings")
		}
		if _, ok := seen[fieldName]; ok {
			return nil, fmt.Errorf("datasets[].loader.%s is defined more than once", fieldName)
		}
		seen[fieldName] = struct{}{}

		switch fieldName {
		case datasetLoaderFieldType, datasetLoaderFieldSubset, datasetLoaderFieldSplit, datasetLoaderFieldRevision, datasetLoaderFieldChecksum:
		default:
			return nil, fmt.Errorf("datasets[].loader contains unknown field %q", fieldName)
		}

		value, ok := field.Value.(string)
		if !ok {
			return nil, fmt.Errorf("datasets[].loader.%s must be a string", fieldName)
		}
		switch fieldName {
		case datasetLoaderFieldType:
			loader.Type = value
		case datasetLoaderFieldSubset:
			loader.Subset = value
		case datasetLoaderFieldSplit:
			loader.Split = value
		case datasetLoaderFieldRevision:
			loader.Revision = value
		case datasetLoaderFieldChecksum:
			loader.Checksum = value
		}

		if fieldName != datasetLoaderFieldType && fieldName != datasetLoaderFieldSplit && strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("datasets[].loader.%s must not be empty", fieldName)
		}
	}

	return loader, nil
}

type FineTuneConfigUnslothSpec struct {
	Packing                   bool    `yaml:"packing"`
	MaxSeqLength              int     `yaml:"maxSeqLength"`
	LoadIn4bit                bool    `yaml:"loadIn4bit"`
	Loss                      string  `yaml:"loss"`
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
	APIVersion string                    `yaml:"apiVersion"`
	Target     string                    `yaml:"target"`
	BaseModel  string                    `yaml:"baseModel"`
	Objective  *rawFineTuneObjectiveSpec `yaml:"objective"`
	Datasets   []Dataset                 `yaml:"datasets"`
	Config     *rawFineTuneConfigSpec    `yaml:"config"`
	Output     *rawFineTuneOutputSpec    `yaml:"output"`
}

type rawFineTuneObjectiveSpec struct {
	Type            *string  `yaml:"type"`
	Beta            *float64 `yaml:"beta"`
	LossType        *string  `yaml:"lossType"`
	MaxPromptLength *int     `yaml:"maxPromptLength"`
}

type rawFineTuneConfigSpec struct {
	Unsloth *rawFineTuneConfigUnslothSpec `yaml:"unsloth"`
}

type rawFineTuneConfigUnslothSpec struct {
	Packing                   *bool    `yaml:"packing"`
	MaxSeqLength              *int     `yaml:"maxSeqLength"`
	LoadIn4bit                *bool    `yaml:"loadIn4bit"`
	Loss                      *string  `yaml:"loss"`
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
	objective := normalizeFineTuneObjective(c.Objective)

	return FineTuneConfig{
		APIVersion: c.APIVersion,
		Target:     c.Target,
		BaseModel:  c.BaseModel,
		Objective:  objective,
		Datasets:   c.Datasets,
		Config: FineTuneConfigSpec{
			Unsloth: normalizeUnslothConfig(rawUnsloth, objective.Type),
		},
		Output: normalizeFineTuneOutput(c.Output),
	}
}

func normalizeFineTuneObjective(c *rawFineTuneObjectiveSpec) FineTuneObjectiveSpec {
	if c == nil {
		return FineTuneObjectiveSpec{Type: utils.ObjectiveSFT}
	}

	objectiveType := valueOrDefault(c.Type, utils.ObjectiveSFT)
	objective := FineTuneObjectiveSpec{
		Type:                      objectiveType,
		Beta:                      valueOrDefault(c.Beta, 0),
		LossType:                  valueOrDefault(c.LossType, ""),
		MaxPromptLength:           valueOrDefault(c.MaxPromptLength, 0),
		betaConfigured:            c.Beta != nil,
		lossTypeConfigured:        c.LossType != nil,
		maxPromptLengthConfigured: c.MaxPromptLength != nil,
	}
	if objectiveType == utils.ObjectiveDPO {
		objective.Beta = valueOrDefault(c.Beta, defaultDPOBeta)
		objective.LossType = valueOrDefault(c.LossType, defaultDPOLossType)
		objective.MaxPromptLength = valueOrDefault(c.MaxPromptLength, defaultDPOMaxPromptLength)
	}
	return objective
}

func normalizeUnslothConfig(c *rawFineTuneConfigUnslothSpec, objectiveType string) FineTuneConfigUnslothSpec {
	if c == nil {
		c = &rawFineTuneConfigUnslothSpec{}
	}
	learningRate := defaultSFTLearningRate
	if objectiveType == utils.ObjectiveDPO {
		learningRate = defaultDPOLearningRate
	}
	return FineTuneConfigUnslothSpec{
		Packing:                   valueOrDefault(c.Packing, false),
		MaxSeqLength:              valueOrDefault(c.MaxSeqLength, defaultMaxSeqLength),
		LoadIn4bit:                valueOrDefault(c.LoadIn4bit, defaultLoadIn4bit),
		Loss:                      valueOrDefault(c.Loss, defaultLoss),
		BatchSize:                 valueOrDefault(c.BatchSize, defaultBatchSize),
		GradientAccumulationSteps: valueOrDefault(c.GradientAccumulationSteps, defaultGradientAccumulationSteps),
		WarmupSteps:               valueOrDefault(c.WarmupSteps, defaultWarmupSteps),
		MaxSteps:                  valueOrDefault(c.MaxSteps, defaultMaxSteps),
		LearningRate:              valueOrDefault(c.LearningRate, learningRate),
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
