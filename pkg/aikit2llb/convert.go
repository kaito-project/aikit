package aikit2llb

import (
	"fmt"
	"net/url"
	"path"
	"strings"

	"github.com/moby/buildkit/client/llb"
	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sozercan/aikit/pkg/aikit/config"
	"github.com/sozercan/aikit/pkg/utils"
)

const (
	debianSlim     = "docker.io/library/debian:12-slim"
	distrolessBase = "gcr.io/distroless/cc-debian12:latest"

	localAIVersion = "v2.6.0"
	localAIRepo    = "https://github.com/mudler/LocalAI"
	cudaVersion    = "12-3"
)

func Aikit2LLB(c *config.Config) (llb.State, *specs.Image) {
	var merge llb.State
	state := llb.Image(debianSlim)
	base := getBaseImage(c)

	state, merge = copyModels(c, base, state)
	state, merge = addLocalAI(c, state, merge)

	// install cuda if runtime is nvidia
	if c.Runtime == utils.RuntimeNVIDIA {
		merge = installCuda(c, state, merge)
	}

	// install opencv and friends if stable diffusion backend is being used
	for b := range c.Backends {
		switch c.Backends[b] {
		case utils.BackendExllama:
		case utils.BackendExllamaV2:
			merge = installExllama(c, state, merge)
		case utils.BackendStableDiffusion:
			merge = installOpenCV(state, merge)
		}
	}

	imageCfg := NewImageConfig(c)
	return merge, imageCfg
}

func getBaseImage(c *config.Config) llb.State {
	for b := range c.Backends {
		switch c.Backends[b] {
		case utils.BackendExllama:
		case utils.BackendExllamaV2:
			return llb.Image(debianSlim)
		case utils.BackendStableDiffusion:
			return llb.Image(debianSlim)
		}
	}
	return llb.Image(distrolessBase)
}

func copyModels(c *config.Config, base llb.State, s llb.State) (llb.State, llb.State) {
	savedState := s

	// create config file if defined
	if c.Config != "" {
		s = s.Run(shf("echo -n \"%s\" > /config.yaml", c.Config)).Root()
	}

	for _, model := range c.Models {
		var opts []llb.HTTPOption
		opts = append(opts, llb.Filename(fileNameFromURL(model.Source)))
		if model.SHA256 != "" {
			digest := digest.NewDigestFromEncoded(digest.SHA256, model.SHA256)
			opts = append(opts, llb.Checksum(digest))
		}

		m := llb.HTTP(model.Source, opts...)

		var modelPath string
		if strings.Contains(model.Name, "/") {
			modelPath = "/models/" + path.Dir(model.Name) + "/" + fileNameFromURL(model.Source)
		} else {
			modelPath = "/models/" + fileNameFromURL(model.Source)
		}

		var copyOpts []llb.CopyOption
		copyOpts = append(copyOpts, &llb.CopyInfo{
			CreateDestPath: true,
		})
		s = s.File(
			llb.Copy(m, fileNameFromURL(model.Source), modelPath, copyOpts...),
			llb.WithCustomName("Copying "+fileNameFromURL(model.Source)+" to "+modelPath), //nolint: goconst
		)

		// create prompt templates if defined
		for _, pt := range model.PromptTemplates {
			if pt.Name != "" && pt.Template != "" {
				s = s.Run(shf("echo -n \"%s\" > /models/%s.tmpl", pt.Template, pt.Name)).Root()
			}
		}
	}
	diff := llb.Diff(savedState, s)
	merge := llb.Merge([]llb.State{base, diff})
	return s, merge
}

func fileNameFromURL(urlString string) string {
	parsedURL, err := url.Parse(urlString)
	if err != nil {
		panic(err)
	}
	return path.Base(parsedURL.Path)
}

func installCuda(c *config.Config, s llb.State, merge llb.State) llb.State {
	cudaKeyringURL := "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
	cudaKeyring := llb.HTTP(cudaKeyringURL)
	s = s.File(
		llb.Copy(cudaKeyring, fileNameFromURL(cudaKeyringURL), "/"),
		llb.WithCustomName("Copying "+fileNameFromURL(cudaKeyringURL)), //nolint: goconst
	)
	s = s.Run(sh("dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb")).Root()
	// running apt-get update twice due to nvidia repo
	s = s.Run(sh("apt-get update && apt-get install -y ca-certificates && apt-get update"), llb.IgnoreCache).Root()

	savedState := s
	// install cuda libraries
	if len(c.Backends) == 0 {
		s = s.Run(shf("apt-get install -y --no-install-recommends libcublas-%[1]s cuda-cudart-%[1]s && apt-get clean", cudaVersion)).Root()
		// using a distroless base image here
		// convert debian package metadata status file to distroless status.d directory
		// clean up apt directories
		s = s.Run(bashf("apt-get install -y --no-install-recommends libcublas-%[1]s cuda-cudart-%[1]s && apt-get clean && mkdir -p /var/lib/dpkg/status.d && description_flag=false; while IFS= read -r line || [[ -n $line ]]; do if [[ $line == Package:* ]]; then pkg_name=$(echo $line | cut -d' ' -f2); elif [[ $line == Maintainer:* ]]; then maintainer=$(echo $line | cut -d' ' -f2-); if [[ $maintainer == 'cudatools <cudatools@nvidia.com>' ]]; then pkg_file=/var/lib/dpkg/status.d/${pkg_name}; echo 'Package: '$pkg_name > $pkg_file; echo $line >> $pkg_file; else pkg_file=''; fi; elif [[ -n $pkg_file ]]; then if [[ $line == Description:* ]]; then description_flag=true; elif [[ $line == '' ]]; then description_flag=false; elif ! $description_flag; then echo $line >> $pkg_file; fi; fi; done < /var/lib/dpkg/status && find /var/lib/dpkg -mindepth 1 ! -regex '^/var/lib/dpkg/status\\.d\\(/.*\\)?' -delete && rm -r /var/lib/apt", cudaVersion)).Root()
	}

	// installing dev dependencies used for exllama
	for b := range c.Backends {
		if c.Backends[b] == utils.BackendExllama || c.Backends[b] == utils.BackendExllamaV2 {
			var exllama2Dep string
			if c.Backends[b] == utils.BackendExllamaV2 {
				exllama2Dep = fmt.Sprintf("libcurand-dev-%[1]s", cudaVersion)
			}
			exllamaDeps := fmt.Sprintf("apt-get install -y --no-install-recommends cuda-cudart-dev-%[1]s cuda-crt-%[1]s libcusparse-dev-%[1]s libcublas-dev-%[1]s libcusolver-dev-%[1]s cuda-nvcc-%[1]s %[2]s && apt-get clean", cudaVersion, exllama2Dep)

			s = s.Run(sh(exllamaDeps)).Root()
		}
	}

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

func installExllama(c *config.Config, s llb.State, merge llb.State) llb.State {
	backend := "exllama"
	exllamaRepo := "https://github.com/turboderp/exllama"
	exllamaTag := "master"
	for b := range c.Backends {
		if c.Backends[b] == utils.BackendExllamaV2 {
			exllamaRepo = "https://github.com/turboderp/exllamav2"
			backend = "exllama2"
			exllamaTag = "v0.0.11"
		}
	}

	savedState := s
	s = s.Run(sh("apt-get update && apt-get install --no-install-recommends -y git ca-certificates python3-pip python3-dev g++ && apt-get clean"), llb.IgnoreCache).Root()

	// clone localai exllama backend only
	s = s.Run(shf("git clone --filter=blob:none --no-checkout %[1]s /tmp/localai/ && cd /tmp/localai && git sparse-checkout init --cone && git sparse-checkout set backend/python/%[2]s && git checkout %[3]s && rm -rf .git", localAIRepo, backend, localAIVersion)).Root()

	// clone exllama to localai exllama backend path and install python dependencies
	s = s.Run(shf("git clone --depth 1 %[1]s --branch %[2]s /tmp/%[3]s && mv /tmp/%[3]s/* /tmp/localai/backend/python/%[3]s && rm -rf /tmp/%[3]s && cd /tmp/localai/backend/python/%[3]s && rm -rf .git && pip3 install grpcio protobuf typing-extensions sympy mpmath setuptools numpy --break-system-packages && pip3 install -r /tmp/localai/backend/python/%[3]s/requirements.txt --break-system-packages", exllamaRepo, exllamaTag, backend)).Root()

	diff := llb.Diff(savedState, s)
	return llb.Merge([]llb.State{merge, diff})
}

func installOpenCV(s llb.State, merge llb.State) llb.State {
	savedState := s
	// adding debian 11 (bullseye) repo due to opencv 4.5 requirement
	s = s.Run(sh("echo 'deb http://deb.debian.org/debian bullseye main' | tee -a /etc/apt/sources.list")).Root()
	// pinning libdap packages to bullseye version due to symbol error
	libdapVersion := "3.20.7-6"
	s = s.Run(shf("apt-get update && mkdir -p /tmp/generated/images && apt-get install -y libopencv-imgcodecs4.5 libgomp1 libdap27=%[1]s libdapclient6v5=%[1]s && apt-get clean", libdapVersion), llb.IgnoreCache).Root()
	diff := llb.Diff(savedState, s)
	merge = llb.Merge([]llb.State{merge, diff})

	sdURL := fmt.Sprintf("https://sertaccdn.azureedge.net/localai/%s/stablediffusion", localAIVersion)
	var opts []llb.HTTPOption
	opts = append(opts, llb.Filename("stablediffusion"))
	opts = append(opts, llb.Chmod(0o755))
	var copyOpts []llb.CopyOption
	copyOpts = append(copyOpts, &llb.CopyInfo{
		CreateDestPath: true,
	})
	sd := llb.HTTP(sdURL, opts...)
	merge = merge.File(
		llb.Copy(sd, "stablediffusion", "/tmp/localai/backend_data/backend-assets/grpc/stablediffusion", copyOpts...),
		llb.WithCustomName("Copying stable diffusion backend"), //nolint: goconst
	)
	return merge
}

func addLocalAI(c *config.Config, s llb.State, merge llb.State) (llb.State, llb.State) {
	savedState := s
	var localAIURL string
	switch c.Runtime {
	case utils.RuntimeNVIDIA:
		localAIURL = fmt.Sprintf("https://sertaccdn.azureedge.net/localai/%s/cuda12/local-ai", localAIVersion)
	case utils.RuntimeCPUAVX2:
		localAIURL = fmt.Sprintf("https://github.com/mudler/LocalAI/releases/download/%s/local-ai-avx2-Linux-x86_64", localAIVersion)
	case utils.RuntimeCPUAVX512:
		localAIURL = fmt.Sprintf("https://github.com/mudler/LocalAI/releases/download/%s/local-ai-avx512-Linux-x86_64", localAIVersion)
	case utils.RuntimeCPUAVX, "":
		localAIURL = fmt.Sprintf("https://github.com/mudler/LocalAI/releases/download/%s/local-ai-avx-Linux-x86_64", localAIVersion)
	}

	var opts []llb.HTTPOption
	opts = append(opts, llb.Filename("local-ai"))
	opts = append(opts, llb.Chmod(0o755))
	localAI := llb.HTTP(localAIURL, opts...)
	s = s.File(
		llb.Copy(localAI, "local-ai", "/usr/bin/local-ai"),
		llb.WithCustomName("Copying "+fileNameFromURL(localAIURL)+" to /usr/bin"), //nolint: goconst
	)

	diff := llb.Diff(savedState, s)
	return s, llb.Merge([]llb.State{merge, diff})
}

func shf(cmd string, v ...interface{}) llb.RunOption {
	return llb.Args([]string{"/bin/sh", "-c", fmt.Sprintf(cmd, v...)})
}

func sh(cmd string) llb.RunOption {
	return llb.Args([]string{"/bin/sh", "-c", cmd})
}

func bashf(cmd string, v ...interface{}) llb.RunOption {
	return llb.Args([]string{"/bin/bash", "-c", fmt.Sprintf(cmd, v...)})
}
