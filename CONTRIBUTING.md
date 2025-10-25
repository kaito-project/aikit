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
git clone https://github.com/kaito-project/aikit.git
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

- Check existing [Issues](https://github.com/kaito-project/aikit/issues) for known problems
- Review the [Documentation](https://kaito-project.github.io/aikit/) for detailed usage instructions
- Create a new issue if you encounter problems or have questions

## Release Process

AIKit uses semantic versioning. Version information is managed in:
- `Makefile`: Update the `VERSION` variable
- `charts/aikit/Chart.yaml`: Update `version` and `appVersion`

The release process is automated through GitHub Actions.

Thank you for contributing to AIKit! 🚀