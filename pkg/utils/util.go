package utils

import (
	"fmt"
	"net/url"
	"path"

	"github.com/moby/buildkit/client/llb"
)

// FileNameFromURL returns the base file name from a URL string. If the URL
// cannot be parsed, it falls back to the base of the raw string rather than
// panicking, so malformed user input surfaces as a build error downstream.
func FileNameFromURL(urlString string) string {
	parsedURL, err := url.Parse(urlString)
	if err != nil {
		return path.Base(urlString)
	}
	return path.Base(parsedURL.Path)
}

// GetBuildArg returns the value of the build arg with the given key, or the
// empty string if it is not set.
func GetBuildArg(opts map[string]string, k string) string {
	if opts != nil {
		if v, ok := opts["build-arg:"+k]; ok {
			return v
		}
	}
	return ""
}

// Sh returns an llb.RunOption that runs the given command with /bin/sh.
func Sh(cmd string) llb.RunOption {
	return llb.Args([]string{"/bin/sh", "-c", cmd})
}

// Shf returns an llb.RunOption that runs the given printf-formatted command with /bin/sh.
func Shf(cmd string, v ...interface{}) llb.RunOption {
	return llb.Args([]string{"/bin/sh", "-c", fmt.Sprintf(cmd, v...)})
}
