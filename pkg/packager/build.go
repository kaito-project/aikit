// Package packager implements the BuildKit gateway frontend used to
// fetch model sources (local, HTTP, Hugging Face) and produce a minimal image
// containing those artifacts for further export (image/oci layout).
package packager

import (
	"context"
	"fmt"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	"github.com/moby/buildkit/frontend/gateway/client"
	v1 "github.com/modelpack/model-spec/specs-go/v1"
)

const (
	localNameContext = "context"
	packModeRaw      = "raw"
)

// BuildModelpack builds a modelpack OCI layout (target packager/modelpack).
func BuildModelpack(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	sessionID := c.BuildOpts().SessionID
	source := getBuildArg(opts, "source")
	if source == "" {
		return nil, fmt.Errorf("source is required for modelpack target")
	}
	hfSecretFlag := getBuildArg(opts, "hf-token")
	exclude := getBuildArg(opts, "exclude")
	packMode := getBuildArg(opts, "layer_packaging")
	if packMode == "" {
		packMode = packModeRaw
	}
	name := determineName(opts)
	refName := determineRefName(opts)
	artifactType := v1.ArtifactTypeModelManifest
	mtManifest := v1.MediaTypeModelConfig
	modelState, err := resolveSourceState(source, sessionID, hfSecretFlag, true, exclude)
	if err != nil {
		return nil, err
	}
	script := generateModelpackScript(packMode, artifactType, mtManifest, name, refName)
	run := llb.Image(bashImage).Run(llb.Args([]string{"bash", "-c", script}), llb.AddMount("/src", modelState, llb.Readonly))
	final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))
	def, err := final.Marshal(ctx, llb.WithCustomName("packager:modelpack"))
	if err != nil {
		return nil, err
	}
	resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, err
	}
	ref, err := resSolve.SingleRef()
	if err != nil {
		return nil, err
	}
	bCfg, _ := createMinimalImageConfig("linux", "amd64")
	out := client.NewResult()
	out.AddMeta(exptypes.ExporterImageConfigKey, bCfg)
	out.SetRef(ref)
	return out, nil
}

// BuildGeneric builds a generic artifact layout (target packager/generic).
func BuildGeneric(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	sessionID := c.BuildOpts().SessionID
	source := getBuildArg(opts, "source")
	if source == "" {
		return nil, fmt.Errorf("source is required for generic target")
	}
	hfSecretFlag := getBuildArg(opts, "hf-token")
	exclude := getBuildArg(opts, "exclude")
	name := determineName(opts)
	refName := determineRefName(opts)
	artifactType := "application/vnd.unknown.artifact.v1"
	debugFlag := getBuildArg(opts, "debug")
	packMode := getBuildArg(opts, "layer_packaging")
	if packMode == "" {
		packMode = packModeRaw
	}
	genericOutputMode := getBuildArg(opts, "generic_output_mode")
	srcState, err := resolveSourceState(source, sessionID, hfSecretFlag, false, exclude)
	if err != nil {
		return nil, err
	}
	if genericOutputMode == "files" {
		// For raw file passthrough, copy directly from the resolved source state root.
		// This avoids relying on an intermediate run mount (which previously caused
		// missing /src path errors in some remote source scenarios).
		final := llb.Scratch().File(llb.Copy(srcState, "/", "/"))
		def, err := final.Marshal(ctx, llb.WithCustomName("packager:generic-files"))
		if err != nil {
			return nil, err
		}
		resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
		if err != nil {
			return nil, err
		}
		ref, err := resSolve.SingleRef()
		if err != nil {
			return nil, err
		}
		bCfg, _ := createMinimalImageConfig("linux", "amd64")
		out := client.NewResult()
		out.AddMeta(exptypes.ExporterImageConfigKey, bCfg)
		out.SetRef(ref)
		return out, nil
	}
	script := generateGenericScript(packMode, artifactType, name, refName, debugFlag == "1")
	run := llb.Image(bashImage).Run(llb.Args([]string{"bash", "-c", script}), llb.AddMount("/src", srcState, llb.Readonly))
	final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))
	def, err := final.Marshal(ctx, llb.WithCustomName("packager:generic"))
	if err != nil {
		return nil, err
	}
	resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, err
	}
	ref, err := resSolve.SingleRef()
	if err != nil {
		return nil, err
	}
	bCfg, _ := createMinimalImageConfig("linux", "amd64")
	out := client.NewResult()
	out.AddMeta(exptypes.ExporterImageConfigKey, bCfg)
	out.SetRef(ref)
	return out, nil
}

func getBuildArg(opts map[string]string, k string) string {
	if opts != nil {
		if v, ok := opts["build-arg:"+k]; ok {
			return v
		}
	}
	return ""
}

// determineRefName returns the reference name to use for index annotations.
// Only uses build-arg:name if present; otherwise returns "latest".
func determineRefName(opts map[string]string) string {
	if n := getBuildArg(opts, "name"); n != "" {
		return n
	}
	// If name not supplied, ref name still "latest" (different semantic than title fallback)
	return "latest"
}

// determineName returns the provided model name (build-arg name) or a fallback.
// Fallback is "aikitmodel" to ensure title annotation isn't empty.
func determineName(opts map[string]string) string {
	if n := getBuildArg(opts, "name"); n != "" {
		return n
	}
	return "aikitmodel"
}
