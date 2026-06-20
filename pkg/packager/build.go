// Package packager implements the BuildKit gateway frontend used to
// fetch model sources (local, HTTP, Hugging Face) and produce a minimal image
// containing those artifacts for further export (image/oci layout).
package packager

import (
	"context"
	"fmt"
	"sort"

	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	"github.com/moby/buildkit/frontend/gateway/client"
	v1 "github.com/modelpack/model-spec/specs-go/v1"
)

const (
	localNameContext    = "context"
	packModeRaw         = "raw"
	defaultPlatformOS   = "linux"
	defaultPlatformArch = "amd64"
)

// buildConfig holds common build parameters extracted from BuildKit options.
type buildConfig struct {
	source            string
	exclude           string
	packMode          string
	name              string
	refName           string
	sessionID         string
	genericOutputMode string
	debug             bool
}

// parseBuildConfig extracts and validates build configuration from BuildKit options.
func parseBuildConfig(opts map[string]string, sessionID string, isModelpack bool) (*buildConfig, error) {
	cfg := &buildConfig{
		source:    utils.GetBuildArg(opts, "source"),
		exclude:   utils.GetBuildArg(opts, "exclude"),
		packMode:  utils.GetBuildArg(opts, "layer_packaging"),
		name:      determineName(opts),
		refName:   determineRefName(opts),
		sessionID: sessionID,
		debug:     utils.GetBuildArg(opts, "debug") == "1",
	}

	if cfg.source == "" {
		target := "generic"
		if isModelpack {
			target = "modelpack"
		}
		return nil, fmt.Errorf("source is required for %s target", target)
	}

	if cfg.packMode == "" {
		cfg.packMode = packModeRaw
	}

	if !isModelpack {
		cfg.genericOutputMode = utils.GetBuildArg(opts, "generic_output_mode")
	}

	return cfg, nil
}

// solveAndBuildResult is a helper that marshals an LLB state, solves it,
// and constructs a client.Result with the appropriate image config.
// This eliminates the repeated marshal→solve→getRef→createConfig→buildResult pattern.
func solveAndBuildResult(ctx context.Context, c client.Client, state llb.State, customName string) (*client.Result, error) {
	def, err := state.Marshal(ctx, llb.WithCustomName(customName))
	if err != nil {
		return nil, fmt.Errorf("failed to marshal %s LLB definition: %w", customName, err)
	}

	resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, fmt.Errorf("failed to solve %s build: %w", customName, err)
	}

	ref, err := resSolve.SingleRef()
	if err != nil {
		return nil, fmt.Errorf("failed to get %s result reference: %w", customName, err)
	}

	bCfg, err := createMinimalImageConfig(defaultPlatformOS, defaultPlatformArch)
	if err != nil {
		return nil, fmt.Errorf("failed to create image config: %w", err)
	}

	out := client.NewResult()
	out.AddMeta(exptypes.ExporterImageConfigKey, bCfg)
	out.SetRef(ref)
	return out, nil
}

// BuildModelpack builds a modelpack OCI layout (target packager/modelpack).
func BuildModelpack(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	sessionID := c.BuildOpts().SessionID

	cfg, err := parseBuildConfig(opts, sessionID, true)
	if err != nil {
		return nil, err
	}

	modelState, err := resolveSourceState(cfg.source, cfg.sessionID, true, cfg.exclude)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve modelpack source %q: %w", cfg.source, err)
	}

	artifactType := v1.ArtifactTypeModelManifest
	mtManifest := v1.MediaTypeModelConfig
	env := modelpackScriptEnv(cfg.packMode, artifactType, mtManifest, cfg.name, cfg.refName)

	runOpts := []llb.RunOption{
		llb.Args([]string{"bash", "-c", modelpackScript}),
		llb.AddMount("/src", modelState, llb.Readonly),
	}
	runOpts = append(runOpts, envRunOptions(env)...)
	run := llb.Image(bashImage).Run(runOpts...)
	final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))

	return solveAndBuildResult(ctx, c, final, "packager:modelpack")
}

// BuildGeneric builds a generic artifact layout (target packager/generic).
func BuildGeneric(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	sessionID := c.BuildOpts().SessionID

	cfg, err := parseBuildConfig(opts, sessionID, false)
	if err != nil {
		return nil, err
	}

	srcState, err := resolveSourceState(cfg.source, cfg.sessionID, false, cfg.exclude)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve generic source %q: %w", cfg.source, err)
	}

	if cfg.genericOutputMode == "files" {
		// For raw file passthrough, copy directly from the resolved source state root.
		// This avoids relying on an intermediate run mount (which previously caused
		// missing /src path errors in some remote source scenarios).
		final := llb.Scratch().File(llb.Copy(srcState, "/", "/"))
		return solveAndBuildResult(ctx, c, final, "packager:generic-files")
	}

	artifactType := "application/vnd.unknown.artifact.v1"
	env := genericScriptEnv(cfg.packMode, artifactType, cfg.name, cfg.refName, cfg.debug)

	runOpts := []llb.RunOption{
		llb.Args([]string{"bash", "-c", genericScript}),
		llb.AddMount("/src", srcState, llb.Readonly),
	}
	runOpts = append(runOpts, envRunOptions(env)...)
	run := llb.Image(bashImage).Run(runOpts...)
	final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))

	return solveAndBuildResult(ctx, c, final, "packager:generic")
}

// envRunOptions converts an environment map into a deterministically ordered
// slice of llb.RunOption values (sorted by key so the produced LLB is stable).
func envRunOptions(env map[string]string) []llb.RunOption {
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	opts := make([]llb.RunOption, 0, len(keys))
	for _, k := range keys {
		opts = append(opts, llb.AddEnv(k, env[k]))
	}
	return opts
}

func getBuildArg(opts map[string]string, k string) string {
	return utils.GetBuildArg(opts, k)
}

// determineRefName returns the reference name to use for index annotations.
// Only uses build-arg:name if present; otherwise returns "latest".
func determineRefName(opts map[string]string) string {
	if n := utils.GetBuildArg(opts, "name"); n != "" {
		return n
	}
	// If name not supplied, ref name still "latest" (different semantic than title fallback)
	return "latest"
}

// determineName returns the provided model name (build-arg name) or a fallback.
// Fallback is "aikitmodel" to ensure title annotation isn't empty.
func determineName(opts map[string]string) string {
	if n := utils.GetBuildArg(opts, "name"); n != "" {
		return n
	}
	return "aikitmodel"
}
