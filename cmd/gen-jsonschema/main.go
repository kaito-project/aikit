// Command gen-jsonschema generates the JSON schema for aikitfile documents.
//
// It reflects the two root config types (InferenceConfig and FineTuneConfig)
// into a single JSON schema whose root is a oneOf over both runtime-valid
// shapes, suitable for editor autocomplete and validation. Field names come
// from the yaml struct tags (aikit parses with yaml.v2 and has no json tags),
// descriptions come from the Go doc comments on the spec structs, and static
// validation rules are imported from the config semantics and pkg/utils so the
// schema cannot drift from the validators. The result is written to
// docs/aikitfile.schema.json.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/invopop/jsonschema"
	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/pkg/errors"
)

// modulePath is the Go module path; it must match the module line in go.mod and
// is used to resolve package import paths when extracting doc comments.
const modulePath = "github.com/kaito-project/aikit"

// configPkgDir is the module-root-relative directory of the config package whose
// doc comments become schema descriptions.
const configPkgDir = "pkg/aikit/config"

// outputPath is the module-root-relative path the generated schema is written to.
const outputPath = "docs/aikitfile.schema.json"

// sha256Pattern mirrors the optional checksum accepted by
// config.InferenceConfig.Validate: absent and empty values are allowed, while a
// non-empty value must be exactly 64 lowercase hexadecimal characters.
const sha256Pattern = `^$|^[a-f0-9]{64}$`

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "gen-jsonschema:", err)
		os.Exit(1)
	}
}

func run() error {
	root, err := moduleRoot()
	if err != nil {
		return err
	}
	// AddGoComments resolves package import paths by joining the module path with
	// the directory it walks, so the generator must run from the module root.
	if err := os.Chdir(root); err != nil {
		return errors.Wrap(err, "chdir to module root")
	}

	reflector := newReflector()
	if err := reflector.AddGoComments(modulePath, configPkgDir); err != nil {
		return errors.Wrap(err, "extract go comments")
	}

	schema, err := buildSchema(reflector)
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return errors.Wrap(err, "marshal schema")
	}
	data = append(data, '\n')

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o750); err != nil {
		return errors.Wrap(err, "create output directory")
	}
	if err := os.WriteFile(outputPath, data, 0o600); err != nil {
		return errors.Wrap(err, "write schema")
	}

	fmt.Println("wrote", filepath.Join(root, outputPath))
	return nil
}

func newReflector() *jsonschema.Reflector {
	return &jsonschema.Reflector{
		// aikit parses with yaml.v2 and has no json tags, so read field names from
		// the yaml struct tag instead of the default json tag.
		FieldNameTag: "yaml",
		// Suppress the reflector's default of requiring every non-omitempty field.
		// buildSchema adds only the fields required by the runtime validators.
		// Unknown keys are still rejected because additionalProperties defaults to
		// false.
		RequiredFromJSONSchemaTags: true,
	}
}

// buildSchema reflects both root config types and merges them into a single
// schema whose root selects exactly one of the two shapes.
func buildSchema(r *jsonschema.Reflector) (*jsonschema.Schema, error) {
	inference := r.Reflect(&config.InferenceConfig{})
	finetune := r.Reflect(&config.FineTuneConfig{})

	defs := jsonschema.Definitions{}
	for name, def := range inference.Definitions {
		defs[name] = def
	}
	for name, def := range finetune.Definitions {
		defs[name] = def
	}

	if err := applyEnums(defs); err != nil {
		return nil, err
	}
	if err := applyValidationConstraints(defs); err != nil {
		return nil, err
	}

	return &jsonschema.Schema{
		Version: jsonschema.Version,
		Title:   "aikitfile",
		OneOf: []*jsonschema.Schema{
			{Ref: "#/$defs/InferenceConfig"},
			{Ref: "#/$defs/FineTuneConfig"},
		},
		Definitions: defs,
	}, nil
}

// applyValidationConstraints mirrors the required-field, collection-size, and
// scalar rules that determine whether the build pipeline accepts a document.
//
// NewFromBytes defaults to inference unless it sees a non-empty baseModel or
// datasets list. A valid finetune config always has exactly one dataset, so
// requiring that runtime-valid shape keeps the root oneOf branches disjoint:
// apiVersion-only documents remain valid inference configs, while finetune
// documents have the positive signal the parser uses to select FineTuneConfig.
// The target stays optional because getAikitfileConfig injects the requested
// target, defaulting to unsloth, after parsing and before validation.
func applyValidationConstraints(defs jsonschema.Definitions) error {
	if err := setRequired(defs, "InferenceConfig", "apiVersion"); err != nil {
		return err
	}
	if err := setArrayMaxItems(defs, "InferenceConfig", "backends", 1); err != nil {
		return err
	}
	if err := setRequired(defs, "FineTuneConfig", "apiVersion", "datasets"); err != nil {
		return err
	}
	if err := setArrayItemBounds(defs, "FineTuneConfig", "datasets", 1, 1); err != nil {
		return err
	}
	if err := setRequired(defs, "Dataset", "type"); err != nil {
		return err
	}
	return setStringPattern(defs, "Model", "sha256", sha256Pattern)
}

// applyEnums constrains the discriminator fields to the exact value sets the
// validators accept, importing the constants from pkg/utils so the schema and
// the validation logic share a single source of truth.
func applyEnums(defs jsonschema.Definitions) error {
	// Mirror config.InferenceConfig.Validate.
	if err := setEnum(defs, "InferenceConfig", "apiVersion", utils.APIv1alpha1); err != nil {
		return err
	}
	if err := setEnum(defs, "InferenceConfig", "runtime",
		"", utils.RuntimeNVIDIA, utils.RuntimeROCm, utils.RuntimeAppleSilicon); err != nil {
		return err
	}
	if err := setItemsEnum(defs, "InferenceConfig", "backends",
		utils.BackendLlamaCpp, utils.BackendDiffusers, utils.BackendVLLM); err != nil {
		return err
	}

	// Mirror config.FineTuneConfig.Validate.
	if err := setEnum(defs, "FineTuneConfig", "apiVersion", utils.APIv1alpha1); err != nil {
		return err
	}
	if err := setEnum(defs, "FineTuneConfig", "target", utils.TargetUnsloth); err != nil {
		return err
	}
	return setEnum(defs, "Dataset", "type", utils.DatasetAlpaca)
}

// setEnum restricts a scalar property to the given values.
func setEnum(defs jsonschema.Definitions, typeName, property string, values ...any) error {
	prop, err := lookupProperty(defs, typeName, property)
	if err != nil {
		return err
	}
	prop.Enum = values
	return nil
}

// setItemsEnum restricts the elements of an array property to the given values.
func setItemsEnum(defs jsonschema.Definitions, typeName, property string, values ...any) error {
	prop, err := lookupProperty(defs, typeName, property)
	if err != nil {
		return err
	}
	if prop.Items == nil {
		return errors.Errorf("property %q on %q is not an array", property, typeName)
	}
	prop.Items.Enum = values
	return nil
}

// setRequired marks the named properties as required after verifying they
// exist on the reflected definition.
func setRequired(defs jsonschema.Definitions, typeName string, properties ...string) error {
	def, ok := defs[typeName]
	if !ok || def.Properties == nil {
		return errors.Errorf("schema definition %q not found", typeName)
	}
	for _, property := range properties {
		if _, err := lookupProperty(defs, typeName, property); err != nil {
			return err
		}
	}
	def.Required = append([]string(nil), properties...)
	return nil
}

// setArrayMaxItems constrains an array property to the runtime maximum.
func setArrayMaxItems(defs jsonschema.Definitions, typeName, property string, maxItems uint64) error {
	prop, err := lookupProperty(defs, typeName, property)
	if err != nil {
		return err
	}
	if prop.Items == nil {
		return errors.Errorf("property %q on %q is not an array", property, typeName)
	}
	prop.MaxItems = &maxItems
	return nil
}

// setArrayItemBounds constrains an array property to the inclusive item-count
// range used by the runtime validator.
func setArrayItemBounds(defs jsonschema.Definitions, typeName, property string, minItems, maxItems uint64) error {
	prop, err := lookupProperty(defs, typeName, property)
	if err != nil {
		return err
	}
	if prop.Items == nil {
		return errors.Errorf("property %q on %q is not an array", property, typeName)
	}
	prop.MinItems = &minItems
	prop.MaxItems = &maxItems
	return nil
}

// setStringPattern constrains a string property to the provided regular
// expression.
func setStringPattern(defs jsonschema.Definitions, typeName, property, pattern string) error {
	prop, err := lookupProperty(defs, typeName, property)
	if err != nil {
		return err
	}
	if prop.Type != "string" {
		return errors.Errorf("property %q on %q is not a string", property, typeName)
	}
	prop.Pattern = pattern
	return nil
}

// lookupProperty resolves a property schema within a reflected definition.
func lookupProperty(defs jsonschema.Definitions, typeName, property string) (*jsonschema.Schema, error) {
	def, ok := defs[typeName]
	if !ok || def.Properties == nil {
		return nil, errors.Errorf("schema definition %q not found", typeName)
	}
	prop, ok := def.Properties.Get(property)
	if !ok || prop == nil {
		return nil, errors.Errorf("property %q not found on %q", property, typeName)
	}
	return prop, nil
}

// moduleRoot walks up from the working directory to the directory containing
// go.mod, so the generator works no matter where go generate invokes it.
func moduleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", errors.Wrap(err, "get working directory")
	}
	for {
		if _, statErr := os.Stat(filepath.Join(dir, "go.mod")); statErr == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("go.mod not found in any parent directory")
		}
		dir = parent
	}
}
