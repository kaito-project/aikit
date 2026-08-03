package inference

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/solver/pb"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	testExactModelName      = "exact"
	testLocalModelDirectory = "models/weights"
	testLocalModelPath      = "models/model.gguf"
	testLocalModelGlob      = "models/*.safetensors"
	testLocalModelZ         = "models/z.gguf"
)

type inferenceDefinitionOp struct {
	digest digest.Digest
	op     *pb.Op
}

func TestInstallBackendMetadataIsDeterministic(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	cfg := &config.InferenceConfig{Backends: []string{utils.BackendLlamaCpp}}
	base := llb.Image(utils.UbuntuBase, llb.Platform(platform))

	var wantHead digest.Digest
	for i := 0; i < 25; i++ {
		state := installBackend(utils.BackendLlamaCpp, cfg, platform, base, base)
		definition, err := state.Marshal(context.Background())
		if err != nil {
			t.Fatalf("marshal backend definition: %v", err)
		}

		head, err := definition.Head()
		if err != nil {
			t.Fatalf("resolve backend definition head: %v", err)
		}
		if i == 0 {
			wantHead = head
			metadata := findInferenceMkfile(t, definition, "/backends/cpu-llama-cpp/metadata.json")

			var got map[string]string
			if err := json.Unmarshal(metadata.Data, &got); err != nil {
				t.Fatalf("unmarshal backend metadata: %v", err)
			}
			want := map[string]string{
				"alias":       utils.BackendLlamaCpp,
				"name":        cpuLlamaCppBackend,
				"gallery_url": "github:mudler/LocalAI/backend/index.yaml@master",
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("backend metadata = %#v, want %#v", got, want)
			}
			if _, ok := got["installed_at"]; ok {
				t.Fatal("backend metadata unexpectedly contains installed_at")
			}
			continue
		}
		if head != wantHead {
			t.Fatalf("backend definition head changed on conversion %d: got %s, want %s", i, head, wantHead)
		}
	}
}

func TestCopyModelsMaterializesConfigurationWithSingleFileOp(t *testing.T) {
	firstTemplate := "first line\n\"double quotes\" '$HOME' `printf ignored` \\ trailing\n{{.Input}}\n"
	secondTemplate := "replacement\nwith 100% literal content and ${SHELL}\n"
	configBody := "models:\n- name: quoted\n  parameters:\n    prompt: \"$HOME `uname` %s \\\\ value\"\n"
	cfg := &config.InferenceConfig{
		Models: []config.Model{
			{
				Name:   "first",
				Source: "first.gguf",
				PromptTemplates: []config.PromptTemplate{
					{Name: "shared", Template: firstTemplate},
					{Name: "special", Template: secondTemplate},
					{Name: "", Template: "ignored"},
					{Name: "empty", Template: ""},
				},
			},
			{
				Name:   "second",
				Source: "second.gguf",
				PromptTemplates: []config.PromptTemplate{
					{Name: "shared", Template: secondTemplate},
				},
			},
		},
		Config: configBody,
	}
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}

	state, _, err := copyModels(cfg, llb.Scratch(), llb.Scratch(), platform, platform)
	if err != nil {
		t.Fatalf("copy models: %v", err)
	}
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal model definition: %v", err)
	}

	var materializationOps []*pb.FileOp
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		if exec := graphOp.op.GetExec(); exec != nil {
			command := strings.Join(exec.Meta.Args, "\x00")
			if strings.Contains(command, ".tmpl") || strings.Contains(command, "/config.yaml") || strings.Contains(command, "echo -n") {
				t.Fatalf("configuration unexpectedly materialized by shell command %q", command)
			}
		}

		fileOp := graphOp.op.GetFile()
		if fileOp == nil {
			continue
		}
		for _, action := range fileOp.Actions {
			if mkfile := action.GetMkfile(); mkfile != nil && (strings.HasSuffix(mkfile.Path, ".tmpl") || mkfile.Path == "/config.yaml") {
				materializationOps = append(materializationOps, fileOp)
				break
			}
		}
	}

	if len(materializationOps) != 1 {
		t.Fatalf("configuration materialization file op count = %d, want 1", len(materializationOps))
	}
	actions := materializationOps[0].Actions
	if len(actions) != 5 {
		t.Fatalf("configuration materialization action count = %d, want 5", len(actions))
	}

	assertInferenceMkfile(t, actions[0], "/models/shared.tmpl", 0o644, firstTemplate)
	assertInferenceMkfile(t, actions[1], "/models/special.tmpl", 0o644, secondTemplate)
	assertInferenceMkfile(t, actions[2], "/models/shared.tmpl", 0o644, secondTemplate)

	mkdir := actions[3].GetMkdir()
	if mkdir == nil {
		t.Fatalf("action 3 = %#v, want mkdir", actions[3].Action)
	}
	if mkdir.Path != "/configuration" || mkdir.Mode != 0o755 || !mkdir.MakeParents {
		t.Fatalf("configuration mkdir = path %q mode %o parents %v, want path %q mode %o parents true", mkdir.Path, mkdir.Mode, mkdir.MakeParents, "/configuration", 0o755)
	}
	assertInferenceMkfile(t, actions[4], "/config.yaml", 0o644, configBody)
}

func TestModelChangesDoNotInvalidateRuntimeBranches(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	baseConfig := &config.InferenceConfig{
		Runtime:  utils.RuntimeNVIDIA,
		Backends: []string{utils.BackendDiffusers},
		Models: []config.Model{
			{Name: "model", Source: "https://example.com/alpha.safetensors"},
		},
		Config: "model: alpha\n",
	}
	changedConfig := *baseConfig
	changedConfig.Models = []config.Model{
		{Name: "model", Source: "https://example.com/beta.safetensors"},
	}
	changedConfig.Config = "model: beta\n"

	baseDefinition := marshalInferenceConfig(t, baseConfig, platform)
	changedDefinition := marshalInferenceConfig(t, &changedConfig, platform)

	for _, customNamePrefix := range []string{
		"Copying local-ai from OCI artifact to /usr/bin",
		"Installing backend diffusers from ",
		"Creating metadata.json for backend cuda12-diffusers",
	} {
		baseOp := findInferenceOpByCustomNamePrefix(t, baseDefinition, customNamePrefix)
		changedOp := findInferenceOpByCustomNamePrefix(t, changedDefinition, customNamePrefix)
		if baseOp.digest != changedOp.digest {
			t.Fatalf("%q digest changed after a model/config-only change: got %s, want %s", customNamePrefix, changedOp.digest, baseOp.digest)
		}
	}

	for _, definition := range []*llb.Definition{baseDefinition, changedDefinition} {
		assertInferenceExecCommandsExclude(t, definition,
			"grpcio-tools",
			"pip install uv",
			"python3-venv",
			"cuda-keyring",
			"libcublas",
			"cuda-cudart",
			"pciutils",
		)
	}

	configName := "Creating config for platform linux/amd64"
	baseModelBranch := findInferenceOpByCustomNamePrefix(t, baseDefinition, configName)
	changedModelBranch := findInferenceOpByCustomNamePrefix(t, changedDefinition, configName)
	if baseModelBranch.digest == changedModelBranch.digest {
		t.Fatalf("model/config branch digest did not change: %s", baseModelBranch.digest)
	}
}

func TestRunnerDependenciesAndEntrypointRemainSequential(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	cfg := &config.InferenceConfig{
		Backends: []string{utils.BackendLlamaCpp},
		Config:   "model: runtime\n",
	}
	definition := marshalInferenceConfig(t, cfg, platform)

	dependencies := findInferenceExecOp(t, definition, "huggingface-hub=="+runnerHuggingFaceHubVersion)
	dependencyCommand := strings.Join(dependencies.op.GetExec().Meta.Args, "\x00")
	for _, fragment := range []string{
		"--no-cache-dir",
		"--no-compile",
		"rm -rf /var/lib/apt/lists/*",
		"/root/.cache/pip",
	} {
		if !strings.Contains(dependencyCommand, fragment) {
			t.Fatalf("runner dependency command = %q, want %q", dependencyCommand, fragment)
		}
	}

	entrypoint := findInferenceOpByCustomNamePrefix(t, definition, "Creating runner entrypoint script")
	modelsDirectory := findInferenceOpByCustomNamePrefix(t, definition, "Creating /models directory with correct ownership")

	assertInferenceOpInput(t, entrypoint, dependencies.digest)
	assertInferenceOpInput(t, modelsDirectory, entrypoint.digest)
}

func TestSelfContainedPythonBackendsAvoidDuplicateRuntimeLayers(t *testing.T) {
	platform := &specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformAMD64}
	tests := []struct {
		name              string
		backend           string
		wantCompilerLayer bool
	}{
		{name: "Diffusers", backend: utils.BackendDiffusers},
		{name: "vLLM", backend: utils.BackendVLLM, wantCompilerLayer: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			definition := marshalInferenceConfig(t, &config.InferenceConfig{
				Runtime:  utils.RuntimeNVIDIA,
				Backends: []string{tt.backend},
			}, platform)

			assertInferenceExecCommandsExclude(t, definition,
				"huggingface-hub",
				"python3-pip",
				"python3-venv",
				"pip install uv",
				"grpcio-tools",
				"cuda-keyring",
				"libcublas",
				"cuda-cudart",
				"pciutils",
			)

			compilerLayers := 0
			for _, graphOp := range decodeInferenceDefinition(t, definition) {
				if exec := graphOp.op.GetExec(); exec != nil && strings.Contains(strings.Join(exec.Meta.Args, "\x00"), "gcc libc6-dev") {
					compilerLayers++
				}
			}
			wantCompilerLayers := 0
			if tt.wantCompilerLayer {
				wantCompilerLayers = 1
			}
			if compilerLayers != wantCompilerLayers {
				t.Fatalf("compiler layer count = %d, want %d", compilerLayers, wantCompilerLayers)
			}
		})
	}
}

func TestLocalModelFollowPaths(t *testing.T) {
	tests := []struct {
		name    string
		sources []string
		want    []string
	}{
		{
			name:    "exact files are normalized sorted and deduplicated",
			sources: []string{testLocalModelZ, "./models/a.gguf", testLocalModelZ, "models/a.gguf"},
			want:    []string{"models/a.gguf", testLocalModelZ},
		},
		{
			name:    "directory without trailing slash",
			sources: []string{testLocalModelDirectory},
			want:    []string{testLocalModelDirectory},
		},
		{
			name:    "directory with trailing slash",
			sources: []string{testLocalModelDirectory + "/"},
			want:    []string{testLocalModelDirectory},
		},
		{
			name:    "parent traversal is clamped to context root",
			sources: []string{"../models/weights"},
			want:    []string{testLocalModelDirectory},
		},
		{
			name:    "context directory remains a path",
			sources: []string{localModelContextName},
			want:    []string{localModelContextName},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := localModelFollowPaths(tt.sources); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("local model follow paths = %#v, want %#v", got, tt.want)
			}
		})
	}

	fallbackSources := []string{
		"",
		".",
		testLocalModelGlob,
		"models/[literal].gguf",
		"!model.gguf",
		" model.gguf",
		"model.gguf ",
		`models\model.gguf`,
	}
	for _, source := range fallbackSources {
		t.Run("unrestricted "+source, func(t *testing.T) {
			if got := localModelFollowPaths([]string{testLocalModelPath, source}); got != nil {
				t.Fatalf("local model follow paths = %#v, want nil for unrestricted source %q", got, source)
			}
		})
	}
}

func TestCopyModelsUsesSingleSymlinkAwareLocalSource(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	models := []config.Model{
		{Name: testExactModelName, Source: testLocalModelPath},
		{Name: "directory", Source: "weights"},
		{Name: "trailing-directory", Source: "adapters/"},
		{Name: "parent", Source: "../normalized/model.bin"},
		{Name: "context-directory", Source: localModelContextName},
		{Name: "duplicate", Source: testLocalModelPath},
		{Name: "remote", Source: "https://example.com/remote.gguf"},
	}
	definition := marshalCopiedModels(t, &config.InferenceConfig{Models: models}, platform)

	localSources := findInferenceLocalContextOps(t, definition)
	if len(localSources) != 1 {
		t.Fatalf("local context source op count = %d, want 1", len(localSources))
	}
	localSource := localSources[0]
	if _, ok := localSource.op.GetSource().Attrs[pb.AttrIncludePatterns]; ok {
		t.Fatal("local context source unexpectedly contains include patterns")
	}

	var followPaths []string
	encodedFollowPaths, ok := localSource.op.GetSource().Attrs[pb.AttrFollowPaths]
	if !ok {
		t.Fatal("local context source does not contain symlink-aware follow paths")
	}
	if err := json.Unmarshal([]byte(encodedFollowPaths), &followPaths); err != nil {
		t.Fatalf("unmarshal local context follow paths: %v", err)
	}
	wantFollowPaths := []string{"adapters", localModelContextName, testLocalModelPath, "normalized/model.bin", "weights"}
	if !reflect.DeepEqual(followPaths, wantFollowPaths) {
		t.Fatalf("local context follow paths = %#v, want %#v", followPaths, wantFollowPaths)
	}

	wantCopies := map[string]int{
		"/adapters":             1,
		"/context":              1,
		"/models/model.gguf":    2,
		"/normalized/model.bin": 1,
		"/weights":              1,
	}
	seenCopies := make(map[string]int, len(wantCopies))
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		fileOp := graphOp.op.GetFile()
		if fileOp == nil {
			continue
		}
		for _, action := range fileOp.Actions {
			copyAction := action.GetCopy()
			if copyAction == nil {
				continue
			}
			if _, ok := wantCopies[copyAction.Src]; !ok {
				continue
			}
			if copyAction.AllowWildcard {
				t.Fatalf("copy %q unexpectedly enables wildcard handling", copyAction.Src)
			}
			secondaryInput := int(action.SecondaryInput)
			if secondaryInput < 0 || secondaryInput >= len(graphOp.op.Inputs) {
				t.Fatalf("copy %q secondary input = %d, inputs = %d", copyAction.Src, secondaryInput, len(graphOp.op.Inputs))
			}
			if got := digest.Digest(graphOp.op.Inputs[secondaryInput].Digest); got != localSource.digest {
				t.Fatalf("copy %q source digest = %s, want shared local source %s", copyAction.Src, got, localSource.digest)
			}
			seenCopies[copyAction.Src]++
		}
	}
	if !reflect.DeepEqual(seenCopies, wantCopies) {
		t.Fatalf("local copy counts = %#v, want %#v", seenCopies, wantCopies)
	}

	reversedModels := append([]config.Model(nil), models...)
	for left, right := 0, len(reversedModels)-1; left < right; left, right = left+1, right-1 {
		reversedModels[left], reversedModels[right] = reversedModels[right], reversedModels[left]
	}
	reversedDefinition := marshalCopiedModels(t, &config.InferenceConfig{Models: reversedModels}, platform)
	reversedLocalSources := findInferenceLocalContextOps(t, reversedDefinition)
	if len(reversedLocalSources) != 1 {
		t.Fatalf("reversed local context source op count = %d, want 1", len(reversedLocalSources))
	}
	if got := reversedLocalSources[0].op.GetSource().Attrs[pb.AttrFollowPaths]; got != encodedFollowPaths {
		t.Fatalf("local context follow path encoding depends on model order: got %q, want %q", got, encodedFollowPaths)
	}
}

func TestCopyModelsUnsafeLiteralPathsUseFullContext(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	for _, source := range []string{
		testLocalModelGlob,
		"models/[literal].gguf",
		"!model.gguf",
		" model.gguf",
		"model.gguf ",
		`models\model.gguf`,
	} {
		t.Run(source, func(t *testing.T) {
			cfg := &config.InferenceConfig{Models: []config.Model{
				{Name: "literal", Source: source},
				{Name: testExactModelName, Source: testLocalModelPath},
			}}
			definition := marshalCopiedModels(t, cfg, platform)
			localSources := findInferenceLocalContextOps(t, definition)
			if len(localSources) != 1 {
				t.Fatalf("local context source op count = %d, want 1", len(localSources))
			}
			attrs := localSources[0].op.GetSource().Attrs
			if _, ok := attrs[pb.AttrFollowPaths]; ok {
				t.Fatalf("literal source %q unexpectedly contains follow paths", source)
			}
			if _, ok := attrs[pb.AttrIncludePatterns]; ok {
				t.Fatalf("literal source %q unexpectedly contains include patterns", source)
			}

			wantCopySource := "/" + normalizeLocalModelPath(source)
			foundLiteralCopy := false
			for _, graphOp := range decodeInferenceDefinition(t, definition) {
				if fileOp := graphOp.op.GetFile(); fileOp != nil {
					for _, action := range fileOp.Actions {
						copyAction := action.GetCopy()
						if copyAction != nil && copyAction.Src == wantCopySource {
							foundLiteralCopy = true
							if copyAction.AllowWildcard {
								t.Fatalf("literal copy %q unexpectedly enables wildcard handling", source)
							}
						}
					}
				}
			}
			if !foundLiteralCopy {
				t.Fatalf("literal copy source %q not found", wantCopySource)
			}
		})
	}
}

func TestCopyModelsFullContextSources(t *testing.T) {
	platform := specs.Platform{OS: utils.PlatformLinux, Architecture: utils.PlatformARM64}
	for _, source := range []string{"", "."} {
		t.Run(source, func(t *testing.T) {
			cfg := &config.InferenceConfig{Models: []config.Model{
				{Name: "full", Source: source},
				{Name: testExactModelName, Source: testLocalModelPath},
			}}
			definition := marshalCopiedModels(t, cfg, platform)
			localSources := findInferenceLocalContextOps(t, definition)
			if len(localSources) != 1 {
				t.Fatalf("local context source op count = %d, want 1", len(localSources))
			}
			attrs := localSources[0].op.GetSource().Attrs
			if _, ok := attrs[pb.AttrFollowPaths]; ok {
				t.Fatalf("full-context source %q unexpectedly contains follow paths", source)
			}
			if _, ok := attrs[pb.AttrIncludePatterns]; ok {
				t.Fatalf("full-context source %q unexpectedly contains include patterns", source)
			}

			foundRootCopy := false
			for _, graphOp := range decodeInferenceDefinition(t, definition) {
				if fileOp := graphOp.op.GetFile(); fileOp != nil {
					for _, action := range fileOp.Actions {
						if copyAction := action.GetCopy(); copyAction != nil && copyAction.Src == "/" {
							foundRootCopy = true
						}
					}
				}
			}
			if !foundRootCopy {
				t.Fatalf("full-context source %q does not copy the context root", source)
			}
		})
	}
}

func marshalInferenceConfig(t *testing.T, cfg *config.InferenceConfig, platform *specs.Platform) *llb.Definition {
	t.Helper()

	state, _, err := Aikit2LLB(cfg, platform)
	if err != nil {
		t.Fatalf("convert inference config: %v", err)
	}
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal inference definition: %v", err)
	}
	return definition
}

func marshalCopiedModels(t *testing.T, cfg *config.InferenceConfig, platform specs.Platform) *llb.Definition {
	t.Helper()

	state, _, err := copyModels(cfg, llb.Scratch(), llb.Scratch(), platform, platform)
	if err != nil {
		t.Fatalf("copy models: %v", err)
	}
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("marshal copied models: %v", err)
	}
	return definition
}

func findInferenceOpByCustomNamePrefix(t *testing.T, definition *llb.Definition, prefix string) inferenceDefinitionOp {
	t.Helper()

	var matches []inferenceDefinitionOp
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		metadata, ok := definition.Metadata[graphOp.digest]
		if ok && strings.HasPrefix(metadata.Description["llb.customname"], prefix) {
			matches = append(matches, graphOp)
		}
	}
	if len(matches) != 1 {
		t.Fatalf("ops with custom name prefix %q = %d, want 1", prefix, len(matches))
	}
	return matches[0]
}

func findInferenceExecOp(t *testing.T, definition *llb.Definition, commandFragment string) inferenceDefinitionOp {
	t.Helper()

	var matches []inferenceDefinitionOp
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		if exec := graphOp.op.GetExec(); exec != nil && strings.Contains(strings.Join(exec.Meta.Args, "\x00"), commandFragment) {
			matches = append(matches, graphOp)
		}
	}
	if len(matches) != 1 {
		t.Fatalf("exec ops containing %q = %d, want 1", commandFragment, len(matches))
	}
	return matches[0]
}

func findInferenceLocalContextOps(t *testing.T, definition *llb.Definition) []inferenceDefinitionOp {
	t.Helper()

	var matches []inferenceDefinitionOp
	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		if source := graphOp.op.GetSource(); source != nil && source.Identifier == "local://context" {
			matches = append(matches, graphOp)
		}
	}
	return matches
}

func assertInferenceExecCommandsExclude(t *testing.T, definition *llb.Definition, fragments ...string) {
	t.Helper()

	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		exec := graphOp.op.GetExec()
		if exec == nil {
			continue
		}
		command := strings.Join(exec.Meta.Args, "\x00")
		for _, fragment := range fragments {
			if strings.Contains(command, fragment) {
				t.Fatalf("exec command %q unexpectedly contains %q", command, fragment)
			}
		}
	}
}

func assertInferenceOpInput(t *testing.T, graphOp inferenceDefinitionOp, want digest.Digest) {
	t.Helper()

	for _, input := range graphOp.op.Inputs {
		if digest.Digest(input.Digest) == want {
			return
		}
	}
	t.Fatalf("op %s does not directly depend on %s", graphOp.digest, want)
}

func decodeInferenceDefinition(t *testing.T, definition *llb.Definition) []inferenceDefinitionOp {
	t.Helper()

	ops := make([]inferenceDefinitionOp, 0, len(definition.Def))
	for _, data := range definition.Def {
		op := new(pb.Op)
		if err := op.Unmarshal(data); err != nil {
			t.Fatalf("unmarshal LLB op: %v", err)
		}
		ops = append(ops, inferenceDefinitionOp{digest: digest.FromBytes(data), op: op})
	}
	return ops
}

func findInferenceMkfile(t *testing.T, definition *llb.Definition, path string) *pb.FileActionMkFile {
	t.Helper()

	for _, graphOp := range decodeInferenceDefinition(t, definition) {
		if fileOp := graphOp.op.GetFile(); fileOp != nil {
			for _, action := range fileOp.Actions {
				if mkfile := action.GetMkfile(); mkfile != nil && mkfile.Path == path {
					return mkfile
				}
			}
		}
	}
	t.Fatalf("mkfile action for %q not found", path)
	return nil
}

func assertInferenceMkfile(t *testing.T, action *pb.FileAction, path string, mode int32, data string) {
	t.Helper()

	mkfile := action.GetMkfile()
	if mkfile == nil {
		t.Fatalf("action = %#v, want mkfile for %q", action.Action, path)
	}
	if mkfile.Path != path || mkfile.Mode != mode || string(mkfile.Data) != data {
		t.Fatalf("mkfile = path %q mode %o data %q, want path %q mode %o data %q", mkfile.Path, mkfile.Mode, string(mkfile.Data), path, mode, data)
	}
}
