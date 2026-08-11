package backendcatalogimport

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v2"
)

const (
	maxSourceBytes       = 16 << 20
	maxSourceEntries     = 5000
	maxCapabilities      = 64
	maxSourceStringBytes = 1 << 20
)

type sourceEntry struct {
	Name         string            `yaml:"name" json:"name"`
	Alias        string            `yaml:"alias" json:"alias,omitempty"`
	License      string            `yaml:"license" json:"license,omitempty"`
	Icon         string            `yaml:"icon" json:"icon,omitempty"`
	Description  string            `yaml:"description" json:"description,omitempty"`
	URLs         []string          `yaml:"urls" json:"urls,omitempty"`
	Tags         []string          `yaml:"tags" json:"tags,omitempty"`
	Capabilities map[string]string `yaml:"capabilities" json:"capabilities,omitempty"`
	URI          string            `yaml:"uri" json:"uri,omitempty"`
	Mirrors      []string          `yaml:"mirrors" json:"mirrors,omitempty"`
}

func parseSource(data []byte, pin SourcePin) ([]sourceEntry, error) {
	if err := verifySource(data, pin); err != nil {
		return nil, err
	}

	if err := validateYAMLShape(data); err != nil {
		return nil, err
	}

	var entries []sourceEntry
	if err := yaml.Unmarshal(data, &entries); err != nil {
		return nil, errors.Wrap(err, "parse LocalAI backend catalog")
	}
	if len(entries) == 0 {
		return nil, errors.New("LocalAI backend catalog is empty")
	}
	if len(entries) > maxSourceEntries {
		return nil, fmt.Errorf("LocalAI backend catalog has %d entries; limit is %d", len(entries), maxSourceEntries)
	}

	for index := range entries {
		if err := validateSourceEntry(entries[index], index); err != nil {
			return nil, err
		}
	}

	return collapseSourceDuplicates(entries)
}

func validateYAMLShape(data []byte) error {
	var mappings []yaml.MapSlice
	if err := yaml.Unmarshal(data, &mappings); err != nil {
		return errors.Wrap(err, "parse LocalAI backend catalog structure")
	}
	allowed := map[string]struct{}{
		"name": {}, "alias": {}, "license": {}, "icon": {}, "description": {},
		"urls": {}, "tags": {}, "capabilities": {}, "uri": {}, "mirrors": {}, "<<": {},
	}
	for index, mapping := range mappings {
		seen := make(map[string]struct{}, len(mapping))
		for _, item := range mapping {
			key, ok := item.Key.(string)
			if !ok {
				return fmt.Errorf("source entry %d contains a non-string YAML key", index)
			}
			if _, ok := allowed[key]; !ok {
				return fmt.Errorf("source entry %d contains unknown field %q", index, key)
			}
			if _, exists := seen[key]; exists {
				return fmt.Errorf("source entry %d contains duplicate explicit field %q", index, key)
			}
			seen[key] = struct{}{}
		}
	}

	var capabilityMappings []struct {
		Capabilities yaml.MapSlice `yaml:"capabilities"`
	}
	if err := yaml.Unmarshal(data, &capabilityMappings); err != nil {
		return errors.Wrap(err, "parse LocalAI capability mappings")
	}
	for index, entry := range capabilityMappings {
		seen := make(map[string]struct{}, len(entry.Capabilities))
		for _, item := range entry.Capabilities {
			selector, ok := item.Key.(string)
			if !ok {
				return fmt.Errorf("source entry %d contains a non-string capability selector", index)
			}
			if _, ok := item.Value.(string); !ok {
				return fmt.Errorf("source entry %d capability %q has a non-string target", index, selector)
			}
			if _, exists := seen[selector]; exists {
				return fmt.Errorf("source entry %d contains duplicate capability selector %q", index, selector)
			}
			seen[selector] = struct{}{}
		}
	}

	return nil
}

func verifySource(data []byte, pin SourcePin) error {
	if len(data) == 0 {
		return errors.New("LocalAI backend catalog source is empty")
	}
	if len(data) > maxSourceBytes {
		return fmt.Errorf("LocalAI backend catalog is %d bytes; limit is %d", len(data), maxSourceBytes)
	}
	if pin.Repository == "" || pin.Path == "" || pin.Revision == "" || pin.SHA256 == "" {
		return errors.New("source pin must include repository, path, revision, and sha256")
	}
	if len(pin.Revision) != 40 || strings.ToLower(pin.Revision) != pin.Revision {
		return fmt.Errorf("source revision %q is not a full Git commit", pin.Revision)
	}
	if _, err := hex.DecodeString(pin.Revision); err != nil {
		return fmt.Errorf("source revision %q is not a full Git commit", pin.Revision)
	}
	algorithm, expected, ok := strings.Cut(pin.SHA256, ":")
	if !ok || algorithm != "sha256" || len(expected) != sha256.Size*2 {
		return fmt.Errorf("source sha256 %q is not a lowercase SHA-256", pin.SHA256)
	}
	if _, err := hex.DecodeString(expected); err != nil || strings.ToLower(pin.SHA256) != pin.SHA256 {
		return fmt.Errorf("source sha256 %q is not a lowercase SHA-256", pin.SHA256)
	}

	digest := sha256.Sum256(data)
	actual := hex.EncodeToString(digest[:])
	if actual != expected {
		return fmt.Errorf("source sha256 mismatch: got sha256:%s, want %s", actual, pin.SHA256)
	}

	return nil
}

func validateSourceEntry(entry sourceEntry, index int) error {
	if entry.Name == "" {
		return fmt.Errorf("source entry %d has an empty name", index)
	}
	if strings.TrimSpace(entry.Name) != entry.Name {
		return fmt.Errorf("source entry %q has surrounding whitespace in its name", entry.Name)
	}
	if len(entry.Description) > maxSourceStringBytes {
		return fmt.Errorf("source entry %q description exceeds %d bytes", entry.Name, maxSourceStringBytes)
	}
	if len(entry.Capabilities) > maxCapabilities {
		return fmt.Errorf("source entry %q has %d capabilities; limit is %d", entry.Name, len(entry.Capabilities), maxCapabilities)
	}
	for selector, target := range entry.Capabilities {
		if selector == "" || strings.TrimSpace(selector) != selector {
			return fmt.Errorf("source entry %q has an invalid empty or whitespace-padded selector", entry.Name)
		}
		if target == "" || strings.TrimSpace(target) != target {
			return fmt.Errorf("source entry %q selector %q has an invalid target", entry.Name, selector)
		}
	}

	return nil
}

func collapseSourceDuplicates(entries []sourceEntry) ([]sourceEntry, error) {
	result := make([]sourceEntry, 0, len(entries))
	byName := make(map[string][]byte, len(entries))
	for _, entry := range entries {
		encoded, err := json.Marshal(entry)
		if err != nil {
			return nil, errors.Wrapf(err, "encode source entry %q for duplicate comparison", entry.Name)
		}
		previous, exists := byName[entry.Name]
		if !exists {
			byName[entry.Name] = encoded
			result = append(result, entry)
			continue
		}
		if !bytes.Equal(previous, encoded) {
			return nil, fmt.Errorf("source contains conflicting entries named %q", entry.Name)
		}
	}

	return result, nil
}
