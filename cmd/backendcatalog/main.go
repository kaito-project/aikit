package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/kaito-project/aikit/internal/backendcatalogimport"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/pkg/errors"
)

const defaultCoreRefTemplate = "ghcr.io/kaito-project/aikit/localai:" + backendcatalogimport.LocalAIVersion + "-{architecture}"

type commandConfig struct {
	Source  backendcatalogimport.SourcePin
	Version string
	Stdout  io.Writer
}

func main() {
	config := commandConfig{
		Source:  backendcatalogimport.LocalAIV482Source,
		Version: backendcatalogimport.LocalAIVersion,
		Stdout:  os.Stdout,
	}
	if err := run(context.Background(), os.Args[1:], config); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, arguments []string, config commandConfig) error {
	flags := flag.NewFlagSet("backendcatalog", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	sourcePath := flags.String("source", "", "path to the pinned LocalAI backend/index.yaml")
	outputPath := flags.String("output", "pkg/backendcatalog/catalog.lock.json", "generated catalog path")
	snapshotPath := flags.String("snapshot", "", "offline OCI resolution snapshot (uses crane when omitted)")
	coreRef := flags.String("core-ref", defaultCoreRefTemplate, "LocalAI core OCI reference; supports {os}, {architecture}, and {variant}")
	craneBinary := flags.String("crane", "crane", "crane binary used when --snapshot is omitted")
	check := flags.Bool("check", false, "verify that --output is current without writing it")
	if err := flags.Parse(arguments); err != nil {
		return errors.Wrap(err, "parse flags")
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	if *sourcePath == "" {
		return errors.New("--source is required")
	}
	if *outputPath == "" {
		return errors.New("--output must not be empty")
	}

	source, err := os.ReadFile(*sourcePath)
	if err != nil {
		return errors.Wrapf(err, "read source %q", *sourcePath)
	}
	resolver, err := loadResolver(*snapshotPath, *craneBinary)
	if err != nil {
		return err
	}
	catalog, err := backendcatalogimport.Generate(ctx, source, backendcatalogimport.GenerateOptions{
		Source:          config.Source,
		Version:         config.Version,
		CoreRefTemplate: *coreRef,
		Resolver:        resolver,
	})
	if err != nil {
		return err
	}
	generated, err := backendcatalogimport.Marshal(catalog)
	if err != nil {
		return err
	}
	if _, err := backendcatalog.Parse(generated); err != nil {
		return errors.Wrap(err, "validate generated backend catalog")
	}

	if *check {
		existing, err := os.ReadFile(*outputPath)
		if err != nil {
			return errors.Wrapf(err, "read generated catalog %q", *outputPath)
		}
		if !bytes.Equal(existing, generated) {
			return fmt.Errorf("generated catalog %q is out of date", *outputPath)
		}
		if config.Stdout != nil {
			_, _ = fmt.Fprintf(config.Stdout, "%s is current\n", *outputPath)
		}

		return nil
	}

	if err := writeFileAtomic(*outputPath, generated); err != nil {
		return err
	}
	if config.Stdout != nil {
		_, _ = fmt.Fprintf(config.Stdout, "wrote %s\n", *outputPath)
	}

	return nil
}

func loadResolver(snapshotPath, craneBinary string) (backendcatalogimport.Resolver, error) {
	if snapshotPath == "" {
		return backendcatalogimport.CraneResolver{Binary: craneBinary}, nil
	}
	snapshot, err := os.ReadFile(snapshotPath)
	if err != nil {
		return nil, errors.Wrapf(err, "read resolution snapshot %q", snapshotPath)
	}
	resolver, err := backendcatalogimport.ParseSnapshot(snapshot)
	if err != nil {
		return nil, err
	}

	return resolver, nil
}

func writeFileAtomic(path string, data []byte) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, ".backendcatalog-*")
	if err != nil {
		return errors.Wrapf(err, "create temporary catalog in %q", directory)
	}
	temporaryPath := temporary.Name()
	defer func() {
		_ = os.Remove(temporaryPath)
	}()

	if _, err := temporary.Write(data); err != nil {
		_ = temporary.Close()
		return errors.Wrap(err, "write temporary catalog")
	}
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return errors.Wrap(err, "set temporary catalog permissions")
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return errors.Wrap(err, "sync temporary catalog")
	}
	if err := temporary.Close(); err != nil {
		return errors.Wrap(err, "close temporary catalog")
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return errors.Wrapf(err, "replace generated catalog %q", path)
	}

	return nil
}
