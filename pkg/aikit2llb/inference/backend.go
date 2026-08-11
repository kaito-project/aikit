package inference

import (
	"encoding/json"
	"fmt"
	"path"
	"slices"
	"strings"
	"unicode"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	yaml "gopkg.in/yaml.v2"
)

type backendMetadata struct {
	Alias         string `json:"alias"`
	Name          string `json:"name"`
	GalleryURL    string `json:"gallery_url"`
	Version       string `json:"version"`
	URI           string `json:"uri"`
	Digest        string `json:"digest"`
	GalleryCommit string `json:"gallery_commit"`
	CatalogDigest string `json:"catalog_digest"`
	Artifact      string `json:"artifact"`
	SourceRef     string `json:"source_ref,omitempty"`
	Runtime       string `json:"runtime,omitempty"`
	Status        string `json:"status,omitempty"`
}

// installBackends installs the exact primary and fallback artifacts from a resolved catalog entry.
func installBackends(backend backendcatalog.Resolution, runtime backendcatalog.Runtime, platform specs.Platform, s llb.State, merge llb.State) llb.State {
	merge = installSystemPackages(backend.SystemPackages, s, merge)
	merge = installRuntimeSymlinks(backend.RuntimeSymlinks, s, merge)

	merge = installBackendArtifact(backend, runtime, backend.Backend, true, platform, s, merge)
	for _, fallback := range backend.Fallbacks {
		merge = installBackendArtifact(backend, runtime, fallback, false, platform, s, merge)
	}

	return merge
}

type localAIModelReference struct {
	Parameters struct {
		Model string `yaml:"model"`
	} `yaml:"parameters"`
}

// localAIModelPaths extracts safe relative model paths without rewriting the
// LocalAI configuration. Invalid or non-path references remain untouched.
func localAIModelPaths(rawConfig string) []string {
	return collapseLocalAIModelPaths(parseLocalAIModelPaths(rawConfig))
}

func parseLocalAIModelPaths(rawConfig string) []string {
	if strings.TrimSpace(rawConfig) == "" {
		return nil
	}

	var modelConfigs []localAIModelReference
	if err := yaml.Unmarshal([]byte(rawConfig), &modelConfigs); err != nil {
		var modelConfig localAIModelReference
		if err := yaml.Unmarshal([]byte(rawConfig), &modelConfig); err != nil {
			return nil
		}
		modelConfigs = []localAIModelReference{modelConfig}
	}

	paths := make(map[string]struct{}, len(modelConfigs))
	for _, modelConfig := range modelConfigs {
		modelPath, ok := safeLocalAIModelPath(modelConfig.Parameters.Model)
		if ok {
			paths[modelPath] = struct{}{}
		}
	}

	result := make([]string, 0, len(paths))
	for modelPath := range paths {
		result = append(result, modelPath)
	}
	slices.Sort(result)
	return result
}

func collapseLocalAIModelPaths(modelPaths []string) []string {
	if len(modelPaths) == 0 {
		return nil
	}

	aliases := make([]string, 0, len(modelPaths))
	for _, modelPath := range modelPaths {
		covered := false
		for _, alias := range aliases {
			if strings.HasPrefix(modelPath, alias+"/") {
				covered = true
				break
			}
		}
		if !covered {
			aliases = append(aliases, modelPath)
		}
	}
	return aliases
}

func safeLocalAIModelPath(modelPath string) (string, bool) {
	if modelPath != strings.TrimSpace(modelPath) || strings.Contains(modelPath, `\`) || strings.Contains(modelPath, "://") {
		return "", false
	}
	for _, r := range modelPath {
		if unicode.IsControl(r) {
			return "", false
		}
	}

	cleaned := path.Clean(modelPath)
	if cleaned == "." || cleaned == ".." || path.IsAbs(cleaned) || strings.HasPrefix(cleaned, "../") {
		return "", false
	}
	return cleaned, true
}

// localAIModelDirectories returns only configured model paths that AIKit can
// prove are materialized directories. File paths and opaque source layouts are
// left to LocalAI's ModelFile handling.
func localAIModelDirectories(rawConfig string, models []config.Model) []string {
	materialized := materializedModelDirectories(models)
	if len(materialized) == 0 {
		return nil
	}

	var aliases []string
	for _, modelPath := range parseLocalAIModelPaths(rawConfig) {
		if _, ok := materialized[modelPath]; ok {
			aliases = append(aliases, modelPath)
		}
	}
	return collapseLocalAIModelPaths(aliases)
}

func materializedModelDirectories(models []config.Model) map[string]struct{} {
	directories := make(map[string]struct{})
	for _, model := range models {
		if strings.HasPrefix(model.Source, "huggingface://") {
			spec, err := ParseHuggingFaceSpec(model.Source)
			if err == nil && spec.SubPath == "" {
				addModelDirectory(directories, path.Join(spec.Namespace, spec.Model))
			}
			continue
		}

		if !strings.HasPrefix(model.Source, "http://") && !strings.HasPrefix(model.Source, "https://") {
			continue
		}
		modelName, ok := safeLocalAIModelPath(model.Name)
		if !ok || !strings.Contains(modelName, "/") {
			continue
		}
		addModelDirectory(directories, path.Dir(modelName))
	}
	return directories
}

func addModelDirectory(directories map[string]struct{}, modelDirectory string) {
	for modelDirectory != "." && modelDirectory != "/" {
		directories[modelDirectory] = struct{}{}
		modelDirectory = path.Dir(modelDirectory)
	}
}

// installBackendModelAliases exposes each configured /models path from every
// backend working directory. LocalAI v4.8.2 passes both Model and ModelFile to
// external backends, but some backends load Model directly as a relative path.
func installBackendModelAliases(backend backendcatalog.Resolution, modelPaths []string, buildPlatform specs.Platform, s llb.State) llb.State {
	if len(modelPaths) == 0 {
		return s
	}

	script := backendModelAliasScript(backend, modelPaths, "/aikit-root")
	helper := orasToolingImage(buildPlatform)
	run := helper.Run(
		utils.Sh(script),
		llb.WithCustomName("Linking baked model directories into backend working directories"),
	)
	return run.AddMount("/aikit-root", s)
}

func backendModelAliasScript(backend backendcatalog.Resolution, modelPaths []string, imageRoot string) string {
	artifacts := make([]backendcatalog.BackendArtifact, 0, 1+len(backend.Fallbacks))
	artifacts = append(artifacts, backend.Backend)
	artifacts = append(artifacts, backend.Fallbacks...)

	seenInstallNames := make(map[string]struct{}, len(artifacts))
	var script strings.Builder
	script.WriteString(`set -eu
fail() {
  printf '%s\n' "$1" >&2
  exit 1
}
require_real_directory() {
  directory=$1
  if [ -L "$directory" ]; then
    fail "Backend model alias path is a symlink: $directory"
  fi
  if [ ! -d "$directory" ]; then
    fail "Backend model alias path is not a directory: $directory"
  fi
}
ensure_real_directory() {
  directory=$1
  if [ -L "$directory" ]; then
    fail "Backend model alias ancestor is a symlink: $directory"
  fi
  if [ -e "$directory" ]; then
    if [ ! -d "$directory" ]; then
      fail "Backend model alias ancestor is not a directory: $directory"
    fi
    return
  fi
  if ! mkdir "$directory"; then
    fail "Failed to create backend model alias ancestor: $directory"
  fi
}
`)
	for _, artifact := range artifacts {
		if _, ok := seenInstallNames[artifact.InstallName]; ok {
			continue
		}
		seenInstallNames[artifact.InstallName] = struct{}{}

		backendDir := path.Join(imageRoot, "backends", artifact.InstallName)
		fmt.Fprintf(&script, "backend_dir=%s\n", quoteShellWord(backendDir))
		script.WriteString("require_real_directory \"$backend_dir\"\n")
		for _, modelPath := range modelPaths {
			alias := path.Join(backendDir, modelPath)
			materialized := path.Join(imageRoot, "models", modelPath)
			target := path.Join("/models", modelPath)
			fmt.Fprintf(&script, "alias_path=%s\n", quoteShellWord(alias))
			fmt.Fprintf(&script, "model_path=%s\n", quoteShellWord(materialized))
			fmt.Fprintf(&script, "model_target=%s\n", quoteShellWord(target))
			script.WriteString("require_real_directory \"$model_path\"\n")

			ancestor := backendDir
			components := strings.Split(modelPath, "/")
			for _, component := range components[:len(components)-1] {
				ancestor = path.Join(ancestor, component)
				fmt.Fprintf(&script, "ancestor=%s\n", quoteShellWord(ancestor))
				script.WriteString("ensure_real_directory \"$ancestor\"\n")
			}

			script.WriteString("if [ -L \"$alias_path\" ]; then\n")
			script.WriteString("  actual_target=$(readlink \"$alias_path\") || fail \"Failed to read backend model alias: $alias_path\"\n")
			script.WriteString("  if [ \"$actual_target\" != \"$model_target\" ]; then\n")
			script.WriteString("    fail \"Backend model alias has unexpected target: $alias_path -> $actual_target\"\n")
			script.WriteString("  fi\n")
			script.WriteString("elif [ -e \"$alias_path\" ]; then\n")
			script.WriteString("  fail \"Backend model alias conflicts with existing path: $alias_path\"\n")
			script.WriteString("elif ! ln -s \"$model_target\" \"$alias_path\"; then\n")
			script.WriteString("  fail \"Failed to create backend model alias: $alias_path\"\n")
			script.WriteString("fi\n")
		}
	}

	return script.String()
}

func quoteShellWord(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func installSystemPackages(packages []string, s llb.State, merge llb.State) llb.State {
	if len(packages) == 0 {
		return merge
	}

	savedState := s
	command := "apt-get update && apt-get install --no-install-recommends -y " + strings.Join(packages, " ") +
		" && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*"
	s = s.Run(
		utils.Sh(command),
		llb.WithCustomName("Installing catalog system packages: "+strings.Join(packages, ", ")),
		llb.IgnoreCache,
	).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

func installRuntimeSymlinks(symlinks []backendcatalog.RuntimeSymlink, s llb.State, merge llb.State) llb.State {
	if len(symlinks) == 0 {
		return merge
	}

	savedState := s
	var actions *llb.FileAction
	for _, symlink := range symlinks {
		if actions == nil {
			actions = llb.Symlink(symlink.Target, symlink.Path)
		} else {
			actions = actions.Symlink(symlink.Target, symlink.Path)
		}
	}
	s = s.File(actions, llb.WithCustomName("Creating catalog runtime compatibility symlinks"))

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

func installBackendArtifact(
	backend backendcatalog.Resolution,
	runtime backendcatalog.Runtime,
	artifact backendcatalog.BackendArtifact,
	primary bool,
	platform specs.Platform,
	s llb.State,
	merge llb.State,
) llb.State {
	savedState := s
	backendDir := "/backends/" + artifact.InstallName

	backendState := llb.Image(
		artifact.Ref,
		llb.Platform(platform),
		llb.WithCustomName(fmt.Sprintf("Installing backend %s from %s", backend.Family, artifact.Ref)),
	)

	metadata := marshalBackendMetadata(backend, runtime, artifact, primary)

	s = s.File(
		llb.Copy(backendState, "/", backendDir+"/", &llb.CopyInfo{
			CreateDestPath: true,
		}).Mkfile(backendDir+"/metadata.json", 0o644, metadata),
		llb.WithCustomName(fmt.Sprintf("Creating metadata.json for backend %s", artifact.InstallName)),
	)

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

func marshalBackendMetadata(backend backendcatalog.Resolution, runtime backendcatalog.Runtime, artifact backendcatalog.BackendArtifact, primary bool) []byte {
	uri := artifact.Ref
	artifactMetadata := backendMetadata{
		Alias:         backend.Family,
		Name:          artifact.InstallName,
		GalleryURL:    pinnedGalleryURL(backend.Source),
		Version:       backend.Version,
		URI:           uri,
		Digest:        artifactDigest(artifact.Ref),
		GalleryCommit: backend.Source.Revision,
		CatalogDigest: backend.CatalogDigest,
		Artifact:      artifact.Ref,
	}
	if primary {
		artifactMetadata.URI = backend.SourceRef
		artifactMetadata.SourceRef = backend.SourceRef
		artifactMetadata.Runtime = string(runtime)
		artifactMetadata.Status = string(backend.Status)
	}

	metadata, err := json.MarshalIndent(artifactMetadata, "", "  ")
	if err != nil {
		// All fields are strings from a validated catalog, so marshaling cannot fail.
		panic(fmt.Sprintf("marshaling validated backend metadata: %v", err))
	}

	return append(metadata, '\n')
}

func pinnedGalleryURL(source backendcatalog.Source) string {
	const githubRepositoryPrefix = "https://github.com/"

	repository := strings.TrimSuffix(source.Repository, "/")
	if repositoryPath, ok := strings.CutPrefix(repository, githubRepositoryPrefix); ok {
		repositoryPath = strings.TrimSuffix(repositoryPath, ".git")
		location := "github:" + repositoryPath
		if source.Path != "" {
			location += "/" + source.Path
		}
		if source.Revision != "" {
			location += "@" + source.Revision
		}
		return location
	}

	location := repository
	if source.Path != "" {
		location += "/" + source.Path
	}
	if source.Revision != "" {
		location += "@" + source.Revision
	}
	return location
}

func artifactDigest(ref string) string {
	_, digest, _ := strings.Cut(ref, "@")
	return digest
}
