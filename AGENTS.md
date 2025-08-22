AIKit Copilot Coding Agent Instructions
======================================

Trust this file first. Search the repo ONLY if info needed for your task is missing or provably outdated.

Summary / Stack
---------------
AIKit builds minimal container images bundling open AI model artifacts (LLMs, diffusion, etc.) plus an OpenAI‑compatible runtime (LocalAI). Declarative YAML ("aikitfile") -> parsed & validated -> transformed into a BuildKit LLB graph -> multi‑arch Docker image (CPU; optional CUDA; experimental Apple Silicon via Podman). Primary language: Go. Aux: YAML (Helm, specs), Docker, Markdown, a Docusaurus (Node 18+) docs site.

Repo Size & Layout (Key Paths)
------------------------------
Root notable files: `Makefile`, `go.mod`, `go.sum`, `Dockerfile`, `Dockerfile.base*`, `AGENTS.md`, `README.md`, `LICENSE`, `charts/` (Helm), `models/` (distributed model specs), `test/` (CI & sample aikitfiles), `pkg/` (Go source), `cmd/frontend/main.go` (binary entrypoint), `scripts/` (utilities), `website/` (docs). Security/CI: `.github/workflows/*.yml` (CodeQL, dependency review, scorecards; other build/test workflows may exist). Lint config: `.golangci.yaml` (implicit—referenced by `make lint`).

High-Level Architecture Flow
----------------------------
`pkg/aikit/config` (schema + semantic validation) -> `pkg/aikit2llb` (convert spec to BuildKit LLB; CUDA & backend specific layering) -> `pkg/build` (build orchestration, args, validations) -> buildx produces image embedding model + templates consumed at runtime by LocalAI server on `:8080`. Supporting packages: `pkg/utils`, `pkg/version`.

Fast vs Slow Loop
-----------------
Fast (seconds < 1 min): modify Go -> `make test` -> `make lint` (repeat until clean). Slow (minutes): building base/runtime or model images (`make build-aikit`, `make build-test-model`). Only invoke slow steps if task logic explicitly affects build graph, runtime image content, or model spec semantics.

Bootstrap (Always Do Once Per Fresh Environment)
------------------------------------------------
1. Install Go >=1.23. `go version` should show 1.23+.
2. Ensure Docker with buildx: if `docker buildx ls` shows no builder: `docker buildx create --name aikitbuilder --use`.
3. Install golangci-lint v2.1.6 (or newer v2.x) if absent.
4. (Optional) NVIDIA GPU: install nvidia-container-toolkit; verify `docker run --gpus all nvidia/cuda:12.2.0-base nvidia-smi` works.
5. (Docs changes only) Node >=18 for `website/`: `npm ci && npm run build`.

Canonical Development Command Order (Keep This Sequence)
--------------------------------------------------------
1. (If you added/removed deps) `go mod tidy` then verify no unintended diff beyond expected module adds.
2. `make test` (unit tests with race & coverage -> outputs `coverage.txt`).
3. `make lint` (formats + static checks).
4. Apply minimal change sets; repeat 2–3 until clean.
5. Commit.

Model / Image (only if required for the task):
`make build-test-model TEST_FILE=test/aikitfile-llama.yaml`
Run locally (CPU): `make run-test-model`. GPU: `make run-test-model-gpu`. Apple Silicon experimental (Podman): `make run-test-model-applesilicon`.

Key Make Targets
----------------
`make test` (race + coverage) | `make lint` (golangci-lint) | `make build-aikit` (base runtime image) | `make build-test-model` (model image from YAML) | `make run-test-model*` (run container) | `make release-manifest NEWVERSION=vX.Y.Z` (bump version + Helm chart).

Environment Variables (Override When Needed)
--------------------------------------------
`REGISTRY` (default ghcr.io/kaito-project) | `TAG` (default test) | `TEST_FILE` (model spec path) | `TEST_IMAGE_NAME` | `OUTPUT_TYPE` (e.g. type=docker or type=registry) | `PLATFORMS` (default linux/amd64,linux/arm64) | `RUNTIME` (backend specific build arg).

Model Spec Conventions
----------------------
Test models (CI / minimal) live in `test/` named `aikitfile-*.yaml` and use syntax directive `#syntax=aikit:test`. They should stay small & stable. Distributed (pre-made) models live in `models/` named `<model>.yaml` with syntax `#syntax=ghcr.io/kaito-project/aikit/aikit:latest`. Always provide correct artifact `sha256` and required prompt template keys (e.g. `chatMsg`). Updating template key names requires corresponding logic updates—avoid silent renames.

Safe Change Patterns
--------------------
Schema change: edit structs + validation in `pkg/aikit/config`, add/adjust unit tests referencing a minimal test aikitfile. Build graph change: modify `pkg/aikit2llb`; ensure `make test` stays green; only build images if logic depends on runtime layering. Utility change: update `pkg/utils`, re-run fast loop. Avoid adding large artifacts to `test/`.

Validating Runtime (Only If Needed)
-----------------------------------
After `make run-test-model`: open http://localhost:8080/chat OR curl:
```
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-1b-instruct","messages":[{"role":"user","content":"ping"}]}'
```
Expect `.choices[0].message.content` field present.

CI & Security Workflows (Replicate Core Steps Locally)
------------------------------------------------------
Security workflows: CodeQL (Go build analysis), Dependency Review, Scorecards—network egress restricted (allowlists for GitHub, module proxy, select security APIs). Treat build/network failures referencing blocked hosts as environment constraints—do not retry indefinitely. Ensure no new outbound hosts unless justified.

Lint & Formatting Rules
------------------------
`make lint` runs golangci-lint (linters include: errcheck, errorlint, govet, staticcheck, ineffassign, misspell, revive, goconst, gocritic, gosec, etc.) plus formatters (gofmt/gofumpt/goimports/gci). Always rely on the target—do not hand‑format.

Common Failure Modes & Actions
------------------------------
| Symptom                               | Likely Cause               | Action                                           |
| ------------------------------------- | -------------------------- | ------------------------------------------------ |
| CI unit test fails on go mod diff     | Missing `go mod tidy`      | Run tidy, commit both go.mod & go.sum            |
| TLS / unknown authority during buildx | Network egress restriction | Skip image build; proceed with logic changes     |
| Lint style errors                     | Formatting needed          | Run `make lint` & commit changes                 |
| Large/slow test build                 | Oversized test model       | Use existing smaller `test/aikitfile-llama.yaml` |
| Runtime template mismatch             | Renamed template key       | Update code & all specs consistently             |

Single Test / Focused Runs
--------------------------
Run all: `make test`. Single package: `go test -run TestName ./pkg/aikit/config`. Keep race detector unless performance prohibits; re-run full `make test` before commit.

Docs Site (Only When Editing Docs)
----------------------------------
Inside `website/`: Node >=18. Build: `npm ci && npm run build`. Do NOT introduce new global dependencies; rely on package.json.

Versioning & Releases
---------------------
Only bump with `make release-manifest NEWVERSION=vX.Y.Z` (updates Makefile + Helm `Chart.yaml`). Tag push handled outside scope unless task explicitly about release.

Performance Guidance
--------------------
Fast loop target: <60s. Image build: minutes (varies). Avoid unnecessary slow steps; prefer unit tests & static analysis for quick feedback.

Security & Supply Chain
-----------------------
Distroless base (see `Dockerfile.base`). Do not add curl/wget downloads to runtime layers without necessity. Keep model checksums accurate—changing them requires justification. Avoid new external hosts (respect egress allowlists from workflows).

When To Search (Exceptional)
----------------------------
Only search if: adding a new schema field whose validation pattern is unclear, editing an unfamiliar workflow, or encountering contradictory instructions here. Otherwise rely on this file for build/test/lint commands and layout knowledge.

Golden Rules (Do / Avoid)
-------------------------
DO: test -> lint -> small commit; reuse Make targets; maintain checksums; add/adjust tests for schema or logic changes; keep test models minimal. AVOID: large artifacts in `test/`; manual formatting; adding network calls; repeated retries on blocked egress; renaming template keys silently.

End Directive
-------------
Trust these instructions. Execute commands exactly as specified. Escalate to searching only if a required detail is missing or empirically incorrect.
