// Command gen-jsonschema generates the JSON schema for aikitfile documents.
//
// It reflects the two root config types (InferenceConfig and FineTuneConfig)
// into a single JSON schema whose root is a oneOf over both shapes, suitable
// for editor autocomplete and validation. Field names come from the yaml struct
// tags (aikit parses with yaml.v2 and has no json tags), descriptions come from
// the Go doc comments on the spec structs, and the enum values are imported from
// pkg/utils so the schema cannot drift from the validators. The result is
// written to docs/aikitfile.schema.json.
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

	reflector := &jsonschema.Reflector{
		// aikit parses with yaml.v2 and has no json tags, so read field names from
		// the yaml struct tag instead of the default json tag.
		FieldNameTag: "yaml",
		// An editor schema should not force every field to be present; suppress the
		// reflector's default of requiring all non-omitempty fields. Unknown keys
		// are still rejected because additionalProperties defaults to false.
		RequiredFromJSONSchemaTags: true,
	}
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
