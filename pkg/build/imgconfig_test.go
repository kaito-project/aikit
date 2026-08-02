package build

import (
	"encoding/json"
	"reflect"
	"testing"

	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

// TestToDockerImageRoundTrip locks the invariant that converting an OCI image
// spec to the Docker-extended spec expected by dockerui preserves every field
// of the image config. This guards the multi-platform build rewrite: dockerui
// marshals BuildResult.Image as the exported image config, and the conversion
// must not drop anything.
//
// Note: the marshaled byte ordering differs from a bare *specs.Image (the outer
// Config field is declared after the embedded ocispec.Image, so "config" sorts
// after "rootfs"). That ordering is the canonical dockerfile-frontend layout and
// is semantically irrelevant — buildkit re-marshals the config before digesting.
// The invariant that matters is semantic identity, asserted here via round-trip.
func TestToDockerImageRoundTrip(t *testing.T) {
	img := &specs.Image{
		Platform: specs.Platform{Architecture: "amd64", OS: "linux"},
		Config: specs.ImageConfig{
			Entrypoint: []string{"local-ai"},
			Cmd:        []string{"--config-file=/config.yaml"},
			Env:        []string{"PATH=/usr/bin"},
			Labels:     map[string]string{"ai.kaito.aikit.runner": "true"},
			WorkingDir: "/",
		},
	}

	got, err := json.Marshal(toDockerImage(img))
	if err != nil {
		t.Fatalf("marshal DockerOCIImage: %v", err)
	}

	// Round-trip back into an OCI image spec: every field must survive.
	var rt specs.Image
	if err := json.Unmarshal(got, &rt); err != nil {
		t.Fatalf("unmarshal DockerOCIImage JSON into specs.Image: %v", err)
	}
	if !reflect.DeepEqual(*img, rt) {
		t.Errorf("toDockerImage lost data in round-trip:\n original: %+v\nround-trip: %+v", *img, rt)
	}

	// Specifically guard the shadowing bug: if the outer Config were left unset,
	// the marshaled config object would be empty. Assert it carried over.
	var shape struct {
		Config map[string]json.RawMessage `json:"config"`
	}
	if err := json.Unmarshal(got, &shape); err != nil {
		t.Fatalf("unmarshal shape: %v", err)
	}
	if len(shape.Config) == 0 {
		t.Errorf("toDockerImage produced an empty config object (shadowed Config field was not set): %s", got)
	}
}

func TestToDockerImageNil(t *testing.T) {
	if got := toDockerImage(nil); got != nil {
		t.Errorf("toDockerImage(nil) = %v, want nil", got)
	}
}
