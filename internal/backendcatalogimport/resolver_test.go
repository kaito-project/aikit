package backendcatalogimport

import (
	"context"
	"strings"
	"testing"

	"github.com/pkg/errors"
)

func TestSnapshotResolverRejectsConflictingPlatformChildren(t *testing.T) {
	_, err := NewSnapshotResolver(Snapshot{
		SchemaVersion: snapshotSchemaVersion,
		References: []SnapshotReference{{
			SourceRef: fixtureReferenceV1,
			Manifests: []ResolvedManifest{
				{Digest: fixtureDigestA, Platform: Platform{OS: platformLinux, Architecture: architectureAMD64}},
				{Digest: fixtureDigestB, Platform: Platform{OS: platformLinux, Architecture: architectureAMD64}},
			},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "conflicting digests") {
		t.Fatalf("NewSnapshotResolver() error = %v", err)
	}
}

func TestSnapshotResolverIsOfflineExactAndDetached(t *testing.T) {
	resolver, err := NewSnapshotResolver(Snapshot{
		SchemaVersion: snapshotSchemaVersion,
		References: []SnapshotReference{{
			SourceRef: fixtureReferenceV1,
			Manifests: []ResolvedManifest{{
				Digest:   fixtureDigestA,
				Platform: Platform{OS: platformLinux, Architecture: "x86_64"},
			}},
		}},
	})
	if err != nil {
		t.Fatalf("NewSnapshotResolver() error = %v", err)
	}
	first, err := resolver.Resolve(context.Background(), fixtureReferenceV1)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if first[0].Platform.Architecture != architectureAMD64 {
		t.Fatalf("Resolve() architecture = %q", first[0].Platform.Architecture)
	}
	first[0].Digest = "changed"
	second, err := resolver.Resolve(context.Background(), fixtureReferenceV1)
	if err != nil {
		t.Fatalf("Resolve() second error = %v", err)
	}
	if first[0].Digest == second[0].Digest {
		t.Fatal("Resolve() returned mutable internal state")
	}
	if _, err := resolver.Resolve(context.Background(), "registry.example/repo:missing"); err == nil || !strings.Contains(err.Error(), "absent") {
		t.Fatalf("Resolve() missing error = %v", err)
	}
}

func TestSnapshotResolverSupportsExplicitNotFoundRecords(t *testing.T) {
	const sourceRef = "registry.example/repo:v1-missing"
	resolver, err := NewSnapshotResolver(Snapshot{
		SchemaVersion: snapshotSchemaVersion,
		References: []SnapshotReference{{
			SourceRef:  sourceRef,
			ErrorClass: resolutionErrorNotFound,
		}},
	})
	if err != nil {
		t.Fatalf("NewSnapshotResolver() error = %v", err)
	}
	_, err = resolver.Resolve(context.Background(), sourceRef)
	if err == nil {
		t.Fatal("Resolve() unexpectedly succeeded")
	}
	class, classified := resolutionErrorClass(err)
	if !classified || class != resolutionErrorNotFound {
		t.Fatalf("Resolve() error = %v, class = %q, classified = %t", err, class, classified)
	}
}

func TestSnapshotResolverRejectsAmbiguousErrorRecords(t *testing.T) {
	tests := []struct {
		name      string
		reference SnapshotReference
		wantErr   string
	}{
		{
			name:      "empty",
			reference: SnapshotReference{SourceRef: fixtureReferenceV1},
			wantErr:   "neither manifests nor errorClass",
		},
		{
			name: "both",
			reference: SnapshotReference{
				SourceRef:  fixtureReferenceV1,
				Manifests:  []ResolvedManifest{{Digest: fixtureDigestA}},
				ErrorClass: resolutionErrorNotFound,
			},
			wantErr: "both manifests and errorClass",
		},
		{
			name:      "unknown class",
			reference: SnapshotReference{SourceRef: fixtureReferenceV1, ErrorClass: "timeout"},
			wantErr:   "unsupported errorClass",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewSnapshotResolver(Snapshot{SchemaVersion: snapshotSchemaVersion, References: []SnapshotReference{test.reference}})
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("NewSnapshotResolver() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func TestClassifyCraneError(t *testing.T) {
	err := classifyCraneError("registry.example/repo:missing", errors.New("MANIFEST_UNKNOWN: manifest unknown"))
	class, classified := resolutionErrorClass(err)
	if !classified || class != resolutionErrorNotFound {
		t.Fatalf("classifyCraneError() class = %q, classified = %t, error = %v", class, classified, err)
	}

	unclassified := classifyCraneError(fixtureReferenceV1, errors.New("unauthorized"))
	if _, classified := resolutionErrorClass(unclassified); classified {
		t.Fatalf("classifyCraneError() classified unrelated failure: %v", unclassified)
	}
}
