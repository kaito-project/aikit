# Contributing to AIKit

Thank you for your interest in contributing to AIKit! This guide will help you set up your development environment and understand the development workflow.

## Prerequisites

Before you begin, ensure you have the following installed on your development machine:

### Required Tools

- **Go**: Version 1.24.4 or later
  - Install from [golang.org](https://golang.org/dl/)
  - Verify installation: `go version`

- **Docker**: Required for building and testing model images
  - Install from [docker.com](https://docs.docker.com/get-docker/)
  - Verify installation: `docker --version`
  - Ensure Docker daemon is running

- **Git**: For version control
  - Most systems have this pre-installed
  - Verify installation: `git --version`

### Optional but Recommended

- **golangci-lint**: For code linting
  - Install: `go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest`
  - Note: The project uses golangci-lint v2 configuration

- **pre-commit**: For automated code quality checks
  - Install: `pip install pre-commit` or `brew install pre-commit`
  - Setup: `pre-commit install` (after cloning the repository)

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sozercan/aikit.git
cd aikit
```

### 2. Verify Go Dependencies

```bash
go mod download
go mod verify
```

### 3. Set up Pre-commit Hooks (Optional)

```bash
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

## Building AIKit

> [!TIP]
> Build targets default to multi-platform (`linux/amd64,linux/arm64`). For local development, pass your host architecture to speed up builds and avoid multi-platform issues — e.g. `make build-aikit PLATFORMS=linux/amd64`. You should also use the `default` buildx builder (`docker buildx use default`) so that locally built images are available to subsequent builds via the `#syntax=` directive.

### Build the AIKit Binary

```bash
make build-aikit
```

This creates a Docker image with the AIKit binary. You can customize the build with:

```bash
# Build with custom registry and tag
make build-aikit REGISTRY=myregistry TAG=mytag

# Build with custom output type
make build-aikit OUTPUT_TYPE=type=registry
```

**Note**: If you encounter TLS certificate issues during Docker builds (e.g., in sandboxed environments), ensure your Go proxy and Docker environment have proper network access and certificate trust chains configured.

### Build a Test Model

```bash
make build-test-model
```

This builds a test model using the default configuration (`test/aikitfile-llama.yaml`). You can specify a different configuration:

```bash
make build-test-model TEST_FILE=test/aikitfile-phi3.yaml
```

## Testing

### Running Unit Tests

```bash
make test
```

This runs all unit tests with race detection and generates a coverage report.

### Running a Test Model Locally

After building a test model, you can run it locally:

```bash
# CPU-only
make run-test-model

# With GPU support (requires NVIDIA Docker runtime)
make run-test-model-gpu

# Apple Silicon (experimental, requires Podman)
make run-test-model-applesilicon
```

The model will be available at `http://localhost:8080`. You can test it by:

1. **Web UI**: Navigate to `http://localhost:8080/chat`
2. **API**: Send requests to the OpenAI-compatible endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

## Code Quality and Linting

### Running the Linter

```bash
# Install golangci-lint v2 (if not already installed)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run linting
export PATH="$(go env GOPATH)/bin:$PATH"
golangci-lint run -v ./... --timeout 5m
```

Note: The project uses golangci-lint v2 configuration. Ensure you have the correct version installed.

### Code Style Guidelines

The project follows standard Go conventions:

- Use `gofmt` for formatting (automatically handled by the linter)
- Follow effective Go guidelines
- Write tests for new functionality
- Add appropriate documentation for exported functions and types

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write code following the project's style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run unit tests
make test

# Build and test a model locally
make build-test-model
make run-test-model

# Run linting
golangci-lint run -v ./... --timeout 5m
```

### 4. Commit Your Changes

If you have pre-commit hooks installed, they will automatically run. Otherwise, ensure your code passes linting before committing:

```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request through the GitHub interface.

## Testing Different Model Configurations

AIKit supports various model configurations. Test files are located in the `test/` directory:

- `aikitfile-llama.yaml`: GGUF model (default)
- `aikitfile-llama-cuda.yaml`: CUDA-enabled GGUF model
- `aikitfile-hf.yaml`: Hugging Face model
- `aikitfile-unsloth.yaml`: Fine-tuning configuration
- `aikitfile-diffusers.yaml`: Diffusion model for image generation

The Unsloth Python environment is fully resolved in `pkg/finetune/pylock.toml`. After changing `pkg/finetune/requirements.in`, install the required uv version and regenerate the lock with:

```bash
make update-unsloth-lock
```

The resolution cutoff in `scripts/update-unsloth-lock.sh` is intentionally pinned. Advance it deliberately when updating dependencies, then regenerate the lock.

To test a specific configuration:

```bash
make build-test-model TEST_FILE=test/aikitfile-hf.yaml
make run-test-model
```

## Platform-Specific Testing

### Multi-Platform Builds

```bash
make build-test-model PLATFORMS=linux/amd64,linux/arm64
```

### GPU Testing

Ensure you have NVIDIA Docker runtime installed:

```bash
make build-test-model RUNTIME=cuda
make run-test-model-gpu
```

### Apple Silicon Testing

Use Podman with GPU acceleration:

```bash
make run-test-model-applesilicon
```

## Project Structure

- `cmd/`: Command-line interface code
- `pkg/`: Core library code
  - `aikit/config/`: Configuration parsing
  - `aikit2llb/`: BuildKit LLB conversion
  - `build/`: Build logic and validation
  - `utils/`: Utility functions
- `test/`: Test configurations and fixtures
- `models/`: Model-specific configurations
- `charts/`: Kubernetes Helm charts
- `website/`: Documentation website (Docusaurus)

## Getting Help

- Check existing [Issues](https://github.com/sozercan/aikit/issues) for known problems
- Review the [Documentation](https://sozercan.github.io/aikit/) for detailed usage instructions
- Create a new issue if you encounter problems or have questions

## Release Process

AIKit uses semantic versioning. A `v*` tag is a production deployment event, so release tags must never be created or moved manually.

To publish a stable release:

1. Run the **Prepare release** workflow from `main` with a version in `vX.Y.Z` form.
2. Open the generated pull request, wait for its checks (approving them if GitHub prompts), then review and merge it into `release-X.Y`. The pull request updates:
   - `Makefile`: the `VERSION` variable
   - `charts/aikit/Chart.yaml`: `version` and `appVersion`
3. Run the **Publish release** workflow from `main` with the same version and obtain approval for the `prod` environment.
4. The workflow validates the version files, release branch ancestry, and merged release pull request before the release GitHub App creates the protected tag.
5. The tag starts the artifact and runner-image publishing workflows. Each workflow stages signed, run-specific candidates. After the complete workflow succeeds, a trusted default-branch reconciler promotes its first successful attempt to the immutable `vX.Y.Z` tags. It separately selects the highest stable version whose applicable workflow succeeded and updates only the mutable `latest` aliases. The app reconciler also assigns GitHub Latest.
6. If the released major/minor line is newer than `main`, the trusted publish workflow uses the separate release-automation App to open a pull request. This includes recovery releases such as `v0.22.1` when an unusable `v0.22.0` tag must remain immutable.

The publisher preflight permits follow-up fixes on the release branch after the preparation pull request, but the preparation pull request must change both version manifests to the requested version and its merge must remain an ancestor of the tagged commit. The preparation pull request and the exact commit selected for release must each have successful `lint` and `unit-test` workflow runs, and every other latest workflow run for each commit must finish without failure. Push CI intentionally runs for documentation-only follow-up commits so every release candidate has exact-commit evidence.

Patch versions must increase within each `release-X.Y` line. Maintenance releases on an older line remain supported, but they publish only their immutable version tags; each mutable `latest` alias remains on the highest stable release that completed its applicable publisher successfully. App and runner aliases can therefore remain on different versions when only one publisher succeeds. Candidate signatures bind the source commit, workflow run, attempt, and artifact name, and an existing public `vX.Y.Z` image tag may never move to another digest. If any release branch with an existing stable tag is missing, restore it from the correct historical commit first; **Prepare release** will not recreate a previously released line from `main`. A restored branch must descend from the latest stable tag on that line.

Preparation pull requests use the non-bypass release-automation App so their `pull_request` checks run. That App cannot create protected tags; the tag-ruleset bypass remains exclusive to the release App used after `prod` approval.

Do not run `git tag`, `git push origin vX.Y.Z`, or force-update a release tag. Rerun the failed workflow for a transient publication failure. If the release commit must change, prepare a new patch version; never move the existing tag. If `main` advances while **Publish release** is awaiting approval, rerun it so the approved guardrails come from the current `main` revision.

Rerunning **Publish release** for an existing immutable tag is a recovery operation only: it does not recreate the tag or retrigger the tag-push publishing workflows. Rerun a failed artifact or runner-image workflow directly only for tags created through this protected flow; do not rerun a publisher that already completed successfully. Legacy runs do not have attempt-bound candidate tags, so repairing their image aliases requires a separate manual provenance review instead of the reconciler.

### Release repository setup

The protected flow requires these one-time repository settings:

- Install a dedicated release GitHub App on this repository with only **Contents: write** permission.
- Install a separate release-automation GitHub App with **Contents: write** and **Pull requests: write** permissions. It creates preparation and version-sync pull requests. Do not grant this App a tag-ruleset bypass.
- Store `RELEASE_APP_CLIENT_ID` as an environment variable and `RELEASE_APP_PRIVATE_KEY` as an environment secret only in the protected `prod` environment. Require a reviewer on `prod`, disallow administrator bypass, and restrict deployments to `main`. Enable self-review prevention when a second maintainer or reviewer team is available.
- Store `RELEASE_AUTOMATION_APP_CLIENT_ID` as an environment variable and `RELEASE_AUTOMATION_APP_PRIVATE_KEY` as an environment secret in a separate `release-automation` environment. Restrict it to `main` and disallow administrator bypass. It does not need another reviewer because the App cannot create protected tags. The trusted latest reconciler also uses this environment as a deployment-branch boundary for its package and release writes.
- Apply a creation ruleset to `refs/tags/v*`. Remove repository-role and administrator bypasses; grant **Always allow** bypass only to the dedicated release GitHub App.
- Apply a second ruleset to `refs/tags/v*` that blocks updates and deletions with no bypass actors. Keeping this separate prevents the release App from moving a tag after creating it.
- If deletion is ever required for recovery, temporarily changing the no-bypass ruleset must be a separate audited break-glass process.
- Before the next release from any branch created before these guardrails, backport the complete `.github/workflows` directory and the release control scripts. Tag-push workflows run from the tagged commit, not from the current `main` branch. For a new tag, **Publish release** verifies that the complete workflow tree and host-executed release guardrails exactly match the immutable `main` workflow revision being approved and refuses stale release branches.

GitHub App tokens are intentionally used instead of the workflow's default `GITHUB_TOKEN`: tags created with `GITHUB_TOKEN` do not start tag-push workflows, while pull requests created with the release-automation App start the required pull-request checks.

Thank you for contributing to AIKit! 🚀
