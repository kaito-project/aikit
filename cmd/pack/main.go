package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/kaito-project/aikit/pkg/oci/packager"
)

func main() {
	var (
		source       = flag.String("source", "", "source to package (file:///abs/path or hf://org/repo)")
		outDir       = flag.String("out", "", "output directory for OCI layout")
		specStr      = flag.String("spec", string(packager.SpecModelPack), "spec to use: modelpack|generic-oci")
		name         = flag.String("name", "", "human-friendly name (index annotations)")
		artifactType = flag.String("artifact-type", "", "override manifest artifactType (optional)")

		mtManifest = flag.String("mt-manifest-config", "", "override media type for manifest config (optional)")
		mtWeights  = flag.String("mt-weights", "", "override media type for weights layer(s) (optional)")
		mtConfig   = flag.String("mt-config", "", "override media type for config/metadata layer(s) (optional)")
		mtDocs     = flag.String("mt-docs", "", "override media type for docs layer(s) (optional)")
	)
	flag.Parse()

	if *source == "" || *outDir == "" {
		fmt.Fprintf(os.Stderr, "usage: %s --source <uri> --out <dir> [--spec modelpack|generic-oci] [--name <name>] [--artifact-type <type>] [--mt-... overrides]\n", os.Args[0])
		os.Exit(2)
	}

	spec := packager.SpecType(*specStr)
	switch spec {
	case packager.SpecModelPack, packager.SpecGeneric:
	default:
		log.Fatalf("unknown spec: %s", *specStr)
	}

	opts := packager.Options{
		Source:       *source,
		OutputDir:    *outDir,
		ArtifactType: *artifactType,
		Spec:         spec,
		Name:         *name,
		MediaTypes: packager.ModelPackMediaTypes{
			ManifestConfig: *mtManifest,
			LayerWeights:   *mtWeights,
			LayerConfig:    *mtConfig,
			LayerDocs:      *mtDocs,
		},
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	ctx := context.Background()
	res, err := packager.Pack(ctx, opts)
	if err != nil {
		log.Fatalf("pack failed: %v", err)
	}
	fmt.Println("OCI layout written to:", res.LayoutPath)
}
