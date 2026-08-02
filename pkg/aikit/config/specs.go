package config

//go:generate go run github.com/kaito-project/aikit/cmd/gen-jsonschema

import (
	"github.com/pkg/errors"
	yaml "gopkg.in/yaml.v2"
)

// NewFromBytes parses an aikitfile, returning either an InferenceConfig or a
// FineTuneConfig depending on the document's shape.
//
// Two correctness properties matter here:
//
//  1. The config kind is chosen by an explicit, positive signal (presence of
//     finetune-only fields) rather than by relying on one unmarshal attempt
//     failing and falling back to the other. The old fallback approach only
//     worked by accident: a finetune document without a "config:" block would
//     decode cleanly as an (empty) InferenceConfig and be misclassified.
//  2. Decoding is strict (UnmarshalStrict), so a misspelled or unknown field
//     such as "bckends:" is rejected with an error instead of being silently
//     dropped, which would otherwise produce a wrong image with no feedback.
func NewFromBytes(b []byte) (*InferenceConfig, *FineTuneConfig, error) {
	// Sniff a minimal set of finetune-distinctive fields without strictness;
	// this only decides which concrete type to strict-decode into.
	var probe struct {
		BaseModel string    `yaml:"baseModel"`
		Datasets  []Dataset `yaml:"datasets"`
	}
	if err := yaml.Unmarshal(b, &probe); err != nil {
		return nil, nil, errors.Wrap(err, "unmarshal config")
	}

	if probe.BaseModel != "" || len(probe.Datasets) > 0 {
		fineTuneConfig := &FineTuneConfig{}
		if err := yaml.UnmarshalStrict(b, fineTuneConfig); err != nil {
			return nil, nil, errors.Wrap(err, "unmarshal finetune config")
		}
		return nil, fineTuneConfig, nil
	}

	inferenceConfig := &InferenceConfig{}
	if err := yaml.UnmarshalStrict(b, inferenceConfig); err != nil {
		return nil, nil, errors.Wrap(err, "unmarshal inference config")
	}
	return inferenceConfig, nil, nil
}
