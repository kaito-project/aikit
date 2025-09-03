package packagerfrontend

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
)

const (
	localNameContext = "context"
)

// Build is the BuildKit frontend entrypoint for the packager syntax.
// It takes build-arg: source, and optionally spec, then creates an image
// containing the referenced model/artifacts.
func Build(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	sessionID := c.BuildOpts().SessionID

	source := getBuildArg(opts, "source")
	// spec := getBuildArg(opts, "spec") // reserved for future use

	var st llb.State
	switch {
	case source == "" || source == "." || source == "context":
		// Use entire local context
		s := llb.Local(localNameContext,
			llb.SessionID(sessionID),
			llb.SharedKeyHint(localNameContext),
		)
		st = llb.Scratch().File(llb.Copy(s, "/", "/"))

	case strings.HasPrefix(source, "https://") || strings.HasPrefix(source, "http://"):
		http := llb.HTTP(source)
		// Copy the downloaded file(s) into the image root
		st = llb.Scratch().File(llb.Copy(http, "/", "/"))

	case strings.HasPrefix(source, "huggingface://"):
		// Reuse shared parser to support optional branch: huggingface://{namespace}/{repo}/{[branch/]file}
		hfURL, modelFile, err := inference.ParseHuggingFaceURL(source)
		if err != nil {
			return nil, fmt.Errorf("invalid huggingface source: %w", err)
		}
		// Download with a fixed filename and copy into image root preserving the name.
		m := llb.HTTP(hfURL, llb.Filename(modelFile))
		st = llb.Scratch().File(llb.Copy(m, modelFile, "/"+filepath.ToSlash(path.Base(modelFile))))

	default:
		// Treat as a path inside local build context (relative glob)
		include := source
		if strings.HasSuffix(include, "/") {
			include += "**"
		}
		s := llb.Local(localNameContext,
			llb.IncludePatterns([]string{include}),
			llb.SessionID(sessionID),
			llb.SharedKeyHint(localNameContext+":"+include),
		)
		st = llb.Scratch().File(llb.Copy(s, "/", "/"))
	}

	def, err := st.Marshal(ctx)
	if err != nil {
		return nil, err
	}

	res, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, err
	}
	return res, nil
}

func getBuildArg(opts map[string]string, k string) string {
	if opts != nil {
		if v, ok := opts["build-arg:"+k]; ok {
			return v
		}
	}
	return ""
}
