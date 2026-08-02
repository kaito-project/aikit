package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/invopop/jsonschema"
	"github.com/kaito-project/aikit/pkg/aikit/config"
)

const (
	inferenceConfigDefinition = "InferenceConfig"
	finetuneConfigDefinition  = "FineTuneConfig"
)

func TestBuildSchemaMatchesConfigDiscrimination(t *testing.T) {
	t.Parallel()

	schema, err := buildSchema(newReflector())
	if err != nil {
		t.Fatalf("buildSchema() error = %v", err)
	}

	tests := []struct {
		name       string
		document   string
		wantValid  bool
		wantBranch string
	}{
		{
			name:       "minimal inference",
			document:   `{"apiVersion":"v1alpha1"}`,
			wantValid:  true,
			wantBranch: inferenceConfigDefinition,
		},
		{
			name:       "inference fields",
			document:   `{"apiVersion":"v1alpha1","runtime":"cuda","backends":["llama-cpp"]}`,
			wantValid:  true,
			wantBranch: inferenceConfigDefinition,
		},
		{
			name:       "minimal finetune",
			document:   `{"apiVersion":"v1alpha1","target":"unsloth","datasets":[{"type":"alpaca"}]}`,
			wantValid:  true,
			wantBranch: finetuneConfigDefinition,
		},
		{
			name:       "finetune with base model",
			document:   `{"apiVersion":"v1alpha1","target":"unsloth","baseModel":"example/model","datasets":[{"source":"example/data","type":"alpaca"}]}`,
			wantValid:  true,
			wantBranch: finetuneConfigDefinition,
		},
		{
			name:      "missing api version",
			document:  `{}`,
			wantValid: false,
		},
		{
			name:      "finetune target without discriminator",
			document:  `{"apiVersion":"v1alpha1","target":"unsloth"}`,
			wantValid: false,
		},
		{
			name:      "empty datasets does not discriminate as finetune",
			document:  `{"apiVersion":"v1alpha1","target":"unsloth","datasets":[]}`,
			wantValid: false,
		},
		{
			name:      "base model without valid finetune fields",
			document:  `{"apiVersion":"v1alpha1","baseModel":"example/model"}`,
			wantValid: false,
		},
		{
			name:      "dataset type is required",
			document:  `{"apiVersion":"v1alpha1","target":"unsloth","datasets":[{}]}`,
			wantValid: false,
		},
		{
			name:      "only one dataset is supported",
			document:  `{"apiVersion":"v1alpha1","target":"unsloth","datasets":[{"type":"alpaca"},{"type":"alpaca"}]}`,
			wantValid: false,
		},
		{
			name:      "mixed inference and finetune fields",
			document:  `{"apiVersion":"v1alpha1","runtime":"cuda","target":"unsloth","datasets":[{"type":"alpaca"}]}`,
			wantValid: false,
		},
		{
			name:      "unsupported api version",
			document:  `{"apiVersion":"v2"}`,
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var document any
			if err := json.Unmarshal([]byte(tt.document), &document); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}

			matches := matchingRootDefinitions(schema, document)
			gotSchemaValid := len(matches) == 1
			if gotSchemaValid != tt.wantValid {
				t.Errorf("schema validity = %v (matching branches %v), want %v", gotSchemaValid, matches, tt.wantValid)
			}
			if tt.wantBranch != "" && !reflect.DeepEqual(matches, []string{tt.wantBranch}) {
				t.Errorf("matching branches = %v, want [%s]", matches, tt.wantBranch)
			}
			if !tt.wantValid && len(matches) != 0 {
				t.Errorf("invalid document matched branches %v, want none", matches)
			}

			gotConfigKind, gotConfigValid := configValidationResult([]byte(tt.document))
			if gotConfigValid != tt.wantValid {
				t.Errorf("config validity = %v (kind %q), want %v", gotConfigValid, gotConfigKind, tt.wantValid)
			}
			if tt.wantBranch != "" && gotConfigKind != tt.wantBranch {
				t.Errorf("config kind = %q, want %q", gotConfigKind, tt.wantBranch)
			}
		})
	}
}

func TestGeneratedSchemaIsCurrent(t *testing.T) {
	root, err := moduleRoot()
	if err != nil {
		t.Fatalf("moduleRoot() error = %v", err)
	}
	t.Chdir(root)

	reflector := newReflector()
	if err := reflector.AddGoComments(modulePath, configPkgDir); err != nil {
		t.Fatalf("AddGoComments() error = %v", err)
	}
	schema, err := buildSchema(reflector)
	if err != nil {
		t.Fatalf("buildSchema() error = %v", err)
	}
	want, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		t.Fatalf("json.MarshalIndent() error = %v", err)
	}
	want = append(want, '\n')

	got, err := os.ReadFile(filepath.Join(root, outputPath))
	if err != nil {
		t.Fatalf("os.ReadFile() error = %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("generated schema is stale; run go generate ./pkg/aikit/config/...")
	}
}

func configValidationResult(document []byte) (string, bool) {
	inference, finetune, err := config.NewFromBytes(document)
	if err != nil {
		return "", false
	}
	if inference != nil {
		return inferenceConfigDefinition, inference.Validate() == nil
	}
	if finetune != nil {
		return finetuneConfigDefinition, finetune.Validate() == nil
	}
	return "", false
}

func matchingRootDefinitions(root *jsonschema.Schema, document any) []string {
	matches := make([]string, 0, len(root.OneOf))
	for _, branch := range root.OneOf {
		if schemaMatches(branch, root.Definitions, document) {
			matches = append(matches, strings.TrimPrefix(branch.Ref, "#/$defs/"))
		}
	}
	return matches
}

// schemaMatches evaluates the JSON Schema subset emitted by this generator.
// Keeping this local avoids adding a production dependency solely for tests.
func schemaMatches(schema *jsonschema.Schema, defs jsonschema.Definitions, value any) bool {
	if schema.Ref != "" {
		definition, ok := defs[strings.TrimPrefix(schema.Ref, "#/$defs/")]
		return ok && schemaMatches(definition, defs, value)
	}

	if len(schema.OneOf) > 0 {
		matches := 0
		for _, candidate := range schema.OneOf {
			if schemaMatches(candidate, defs, value) {
				matches++
			}
		}
		if matches != 1 {
			return false
		}
	}

	if len(schema.Enum) > 0 {
		matched := false
		for _, candidate := range schema.Enum {
			if reflect.DeepEqual(candidate, value) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}

	switch schema.Type {
	case "":
		return true
	case "object":
		object, ok := value.(map[string]any)
		if !ok {
			return false
		}
		for _, required := range schema.Required {
			if _, ok := object[required]; !ok {
				return false
			}
		}
		for name, propertyValue := range object {
			property, ok := schema.Properties.Get(name)
			if !ok {
				if !allowsAdditionalProperty(schema.AdditionalProperties, defs, propertyValue) {
					return false
				}
				continue
			}
			if !schemaMatches(property, defs, propertyValue) {
				return false
			}
		}
		return true
	case "array":
		array, ok := value.([]any)
		if !ok {
			return false
		}
		if schema.MinItems != nil && uint64(len(array)) < *schema.MinItems {
			return false
		}
		if schema.MaxItems != nil && uint64(len(array)) > *schema.MaxItems {
			return false
		}
		for _, item := range array {
			if schema.Items != nil && !schemaMatches(schema.Items, defs, item) {
				return false
			}
		}
		return true
	case "string":
		_, ok := value.(string)
		return ok
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "integer":
		number, ok := value.(float64)
		return ok && math.Trunc(number) == number
	case "number":
		_, ok := value.(float64)
		return ok
	default:
		return false
	}
}

func allowsAdditionalProperty(schema *jsonschema.Schema, defs jsonschema.Definitions, value any) bool {
	if schema == nil {
		return true
	}
	encoded, err := json.Marshal(schema)
	if err != nil {
		return false
	}
	if bytes.Equal(encoded, []byte("true")) {
		return true
	}
	if bytes.Equal(encoded, []byte("false")) {
		return false
	}
	return schemaMatches(schema, defs, value)
}
