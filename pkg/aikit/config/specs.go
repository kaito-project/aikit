package config

import (
	"github.com/pkg/errors"
	yaml "gopkg.in/yaml.v2"
)

func NewFromBytes(b []byte) (*InferenceConfig, *FineTuneConfig, error) {
	var fields map[string]interface{}
	if err := yaml.Unmarshal(b, &fields); err != nil {
		return nil, nil, errors.Wrap(err, "unmarshal config")
	}

	_, hasBaseModel := fields["baseModel"]
	_, hasDatasets := fields["datasets"]
	if hasBaseModel || hasDatasets {
		fineTuneConfig := &FineTuneConfig{}
		if err := yaml.Unmarshal(b, fineTuneConfig); err != nil {
			return nil, nil, errors.Wrap(err, "unmarshal config")
		}
		return nil, fineTuneConfig, nil
	}

	inferenceConfig := &InferenceConfig{}
	if err := yaml.Unmarshal(b, inferenceConfig); err != nil {
		return nil, nil, errors.Wrap(err, "unmarshal config")
	}
	return inferenceConfig, nil, nil
}
