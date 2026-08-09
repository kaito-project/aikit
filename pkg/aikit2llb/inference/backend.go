package inference

import (
	"encoding/json"
	"fmt"

	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

const defaultBackendName = "llama-cpp"

type backendMetadata struct {
	Alias         string `json:"alias"`
	Name          string `json:"name"`
	GalleryURL    string `json:"gallery_url"`
	GalleryCommit string `json:"gallery_commit"`
	CatalogDigest string `json:"catalog_digest"`
	Artifact      string `json:"artifact"`
	SourceRef     string `json:"source_ref,omitempty"`
	Version       string `json:"version,omitempty"`
	Selector      string `json:"selector,omitempty"`
	Status        string `json:"status,omitempty"`
}

// installBackends installs the exact primary and fallback artifacts from a resolved catalog entry.
func installBackends(backend backendcatalog.Resolution, platform specs.Platform, s llb.State, merge llb.State) llb.State {
	switch backend.DependencyProfile {
	case backendcatalog.DependencyProfileDiffusers:
		merge = installDiffusersDependencies(s, merge)
	case backendcatalog.DependencyProfileVLLM:
		merge = installVLLMDependencies(s, merge)
	case backendcatalog.DependencyProfileNone:
		// No dependency layer is required.
	default:
		panic(fmt.Sprintf("unsupported validated dependency profile %q", backend.DependencyProfile))
	}

	merge = installBackendArtifact(backend, backend.Backend, true, platform, s, merge)
	for _, fallback := range backend.Fallbacks {
		merge = installBackendArtifact(backend, fallback, false, platform, s, merge)
	}

	return merge
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

	artifactMetadata := backendMetadata{
		Alias:         backend.Family,
		Name:          artifact.InstallName,
		GalleryURL:    backend.Source.Repository,
		GalleryCommit: backend.Source.Revision,
		CatalogDigest: backend.CatalogDigest,
		Artifact:      artifact.Ref,
	}
	if primary {
		artifactMetadata.SourceRef = backend.SourceRef
		artifactMetadata.Version = backend.Version
		artifactMetadata.Selector = string(backend.Selector)
		artifactMetadata.Status = string(backend.Status)
	}
	metadata, err := json.MarshalIndent(artifactMetadata, "", "  ")
	if err != nil {
		// All fields are strings from a validated catalog, so marshaling cannot fail.
		panic(fmt.Sprintf("marshaling validated backend metadata: %v", err))
	}
	metadata = append(metadata, '\n')

	s = s.File(
		llb.Copy(backendState, "/", backendDir+"/", &llb.CopyInfo{
			CreateDestPath: true,
		}).Mkfile(backendDir+"/metadata.json", 0o644, metadata),
		llb.WithCustomName(fmt.Sprintf("Creating metadata.json for backend %s", artifact.InstallName)),
	)

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
