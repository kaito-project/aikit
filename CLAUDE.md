# CLAUDE.md

## Build & Test Commands

- `make test` — run all unit tests (`go test -v ./... -race -coverprofile=coverage.txt -covermode=atomic`)
- `make lint` — run linter (`golangci-lint run -v ./... --timeout 5m`)
- `make build-aikit` — build the AIKit Docker image via `docker buildx`
- `make build-test-model` — build a test model image from a YAML aikitfile
- Run `go mod tidy` after changing dependencies; CI verifies `go.mod`/`go.sum` are clean

## Code Style & Formatting

- golangci-lint v2 with formatters: `gofmt`, `gofumpt`, `goimports`, `gci` (import ordering)
- Key linters enforced: `errcheck`, `errorlint`, `gosec`, `govet`, `staticcheck`, `revive`, `goconst`, `gocritic`, `godot`, `forcetypeassert`, `unconvert`, `unused`, `whitespace`, `misspell` (US locale)
- Max line length: 200 characters
- End every comment with a period (enforced by `godot`)
- All files must end with a newline and have no trailing whitespace (pre-commit hooks)

## Commit Conventions

- PR titles must follow conventional commits: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `perf`, `refactor`, `revert`, `style`, `test`
- Pre-commit hooks run: `gitleaks` (secret scanning), `golangci-lint`, `shellcheck`, `typos`

## Go Conventions

- Module path: `github.com/kaito-project/aikit`
- Go 1.24.3 minimum, toolchain go1.26.1
- Use `github.com/pkg/errors` for error wrapping (not `fmt.Errorf` with `%w`)
- Logging via `github.com/sirupsen/logrus`
- YAML parsing via `gopkg.in/yaml.v2`

## Architecture Quick Reference

- `cmd/frontend/` — BuildKit frontend entrypoint
- `pkg/aikit/config/` — aikitfile YAML config structs and parsing
- `pkg/aikit2llb/` — converts aikitfile configs to BuildKit LLB (inference/ and finetune/ subdirs)
- `pkg/build/` — build orchestration and validation
- `pkg/packager/` — OCI artifact packaging following CNCF ModelPack spec
- `models/` — pre-made model YAML configs
- `runners/` — runner definition YAMLs (llama-cpp-cpu, llama-cpp-cuda, vllm-cuda, diffusers-cuda)
- `test/` — test aikitfile YAML fixtures
