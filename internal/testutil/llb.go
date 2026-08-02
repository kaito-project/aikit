// Package testutil provides small shared helpers for tests across the aikit
// packages. It lives in a non-test file so it can be imported from _test.go
// files in other packages.
package testutil

import "github.com/moby/buildkit/client/llb"

// MarshalToString concatenates the marshaled protobuf operations of an LLB
// definition into a single string. Tests use it to assert that an expected
// command or source identifier appears somewhere in the produced LLB graph.
func MarshalToString(def *llb.Definition) string {
	if def == nil {
		return ""
	}
	var combined []byte
	for _, d := range def.ToPB().Def {
		combined = append(combined, d...)
	}
	return string(combined)
}
