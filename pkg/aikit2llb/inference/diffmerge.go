package inference

import "github.com/moby/buildkit/client/llb"

// applyAndMerge runs fn against state s, then folds the resulting filesystem
// delta onto merge. It replaces the repeated savedState/Diff/Merge boilerplate
// that previously appeared in every backend installer, which was easy to get
// wrong (diffing against the wrong saved state, or merging the wrong base).
//
// It returns the advanced state (for callers that chain further operations) and
// the new merge state with the delta applied.
func applyAndMerge(s, merge llb.State, fn func(llb.State) llb.State) (llb.State, llb.State) {
	next := fn(s)
	return next, llb.Merge([]llb.State{merge, llb.Diff(s, next)})
}
