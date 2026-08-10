package inference

import (
	"github.com/kaito-project/aikit/pkg/utils"
	"github.com/moby/buildkit/client/llb"
)

// installParakeetCppDependencies installs FFmpeg for normalizing uploaded audio
// to the 16 kHz mono WAV format consumed by parakeet.cpp.
func installParakeetCppDependencies(s llb.State, merge llb.State) llb.State {
	savedState := s
	s = s.Run(
		utils.Sh("apt-get update && apt-get install --no-install-recommends -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*"),
		llb.WithCustomName("Installing FFmpeg for parakeet.cpp audio conversion"),
		llb.IgnoreCache,
	).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}
