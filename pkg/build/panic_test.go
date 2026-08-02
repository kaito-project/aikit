package build

import (
	"context"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/pkg/aikit/config"
)

func TestBuildImageWithRecoveryConvertsPanicToError(t *testing.T) {
	result, err := buildImageWithRecovery(
		context.Background(),
		nil,
		&config.InferenceConfig{},
		nil,
		nil,
		nil,
	)
	if result != nil {
		t.Fatalf("buildImageWithRecovery() result = %v, want nil", result)
	}
	if err == nil {
		t.Fatal("buildImageWithRecovery() returned no error for a platform-build panic")
	}
	if !strings.Contains(err.Error(), "recovered from panic in frontend") {
		t.Fatalf("buildImageWithRecovery() error = %q, want recovered panic", err)
	}
}
