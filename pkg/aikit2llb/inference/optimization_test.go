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

	state, _, err := copyModels(cfg, llb.Scratch(), llb.Scratch(), platform)
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
