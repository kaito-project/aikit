package backendcatalogimport

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/pkg/errors"
)

func TestCraneResolverPinsTagBeforeInspectingSingleManifest(t *testing.T) {
	const sourceRef = "registry.example/repo:v1"
	immutableRef := "registry.example/repo@" + fixtureDigestA
	var calls []string
	resolver := CraneResolver{
		Binary: "test-crane",
		run: func(_ context.Context, binary string, arguments ...string) ([]byte, error) {
			calls = append(calls, strings.Join(append([]string{binary}, arguments...), " "))
			switch arguments[0] {
			case "digest":
				if arguments[1] != sourceRef {
					t.Fatalf("digest reference = %q", arguments[1])
				}
				return []byte(fixtureDigestA + "\n"), nil
			case "manifest":
				if arguments[1] != immutableRef {
					t.Fatalf("manifest reference = %q", arguments[1])
				}
				return []byte(`{"schemaVersion":2}`), nil
			case "config":
				if arguments[1] != immutableRef {
					t.Fatalf("config reference = %q", arguments[1])
				}
				return []byte(`{"os":"linux","architecture":"amd64"}`), nil
			default:
				t.Fatalf("unexpected crane command %q", arguments[0])
				return nil, nil
			}
		},
	}

	manifests, err := resolver.Resolve(context.Background(), sourceRef)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if len(manifests) != 1 || manifests[0].Digest != fixtureDigestA || manifests[0].Platform.key() != "linux/amd64/" {
		t.Fatalf("Resolve() = %#v", manifests)
	}
	wantCalls := []string{
		"test-crane digest " + sourceRef,
		"test-crane manifest " + immutableRef,
		"test-crane config " + immutableRef,
	}
	if !slices.Equal(calls, wantCalls) {
		t.Fatalf("crane calls = %v, want %v", calls, wantCalls)
	}
}

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
				Platform: Platform{OS: platformLinux, Architecture: architectureX8664},
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
