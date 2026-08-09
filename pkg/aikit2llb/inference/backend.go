package inference

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
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
	Selector      string `json:"selector,omitempty"`
	Status        string `json:"status,omitempty"`
}

// installBackends installs the exact primary and fallback artifacts from a resolved catalog entry.
func installBackends(backend backendcatalog.Resolution, platform specs.Platform, s llb.State, merge llb.State) llb.State {
	merge = installSystemPackages(backend.SystemPackages, s, merge)
	merge = installRuntimeSymlinks(backend.RuntimeSymlinks, s, merge)

	merge = installBackendArtifact(backend, backend.Backend, true, platform, s, merge)
	for _, fallback := range backend.Fallbacks {
		merge = installBackendArtifact(backend, fallback, false, platform, s, merge)
	}

	return merge
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

	metadata := marshalBackendMetadata(backend, artifact, primary)

	s = s.File(
		llb.Copy(backendState, "/", backendDir+"/", &llb.CopyInfo{
			CreateDestPath: true,
		}).Mkfile(backendDir+"/metadata.json", 0o644, metadata),
		llb.WithCustomName(fmt.Sprintf("Creating metadata.json for backend %s", artifact.InstallName)),
	)

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

func marshalBackendMetadata(backend backendcatalog.Resolution, artifact backendcatalog.BackendArtifact, primary bool) []byte {
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
		artifactMetadata.Selector = string(backend.Selector)
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
