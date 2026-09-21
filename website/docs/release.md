---
title: Release Process
---

The release process is as follows:

- Trigger the [release-pr action](https://github.com/kaito-project/aikit/actions/workflows/release-pr.yaml) with the version to release to create a release PR. Merge the PR to the applicable `release-X.Y` branch.

- Tag the `release-X.Y` branch with a version number that's semver compliant (vMAJOR.MINOR.PATCH), and push the tag to GitHub.

```bash
git tag v0.1.0
git push origin v0.1.0
```

- GitHub Actions will automatically build the AIKit image and push the versioned and `latest` tag to GitHub Container Registry (GHCR) using [release action](https://github.com/kaito-project/aikit/actions/workflows/release.yaml).

## Predefined models

[`models/catalog.json`](https://github.com/kaito-project/aikit/blob/main/models/catalog.json) is the active preset list for both [publishing](https://github.com/kaito-project/aikit/actions/workflows/update-models.yaml) and [weekly patching](https://github.com/kaito-project/aikit/actions/workflows/patch-models.yaml). Each entry maps a recipe filename to its image name, canonical tag, optional alias, and optional platform restriction. Retired recipes and tags must be removed from this catalog together.

After releasing the frontend:

1. Trigger `update-models` with `staging: true`. Set `models` to a JSON array of recipe IDs, such as `["qwen-3.5-4b", "gpt-oss-20b"]`, or `[]` for all active presets. Unknown and retired IDs are rejected. The default runtime list is `["cuda", "applesilicon"]`; FLUX.2 is CUDA-only and builds for AMD64 only.
2. Validate the staged images on appropriate hardware. CUDA images use `ghcr.io/kaito-project/aikit/test/`; Apple Silicon images use `ghcr.io/kaito-project/aikit/test/applesilicon/`. Check model loading, `/v1/models`, chat and streaming responses, tool calls, reasoning output, and image generation where applicable. Check CPU fallback for text images and the advertised architectures. Stage the Qwen 3.5 4B quickstart before directing users to it.
3. Confirm that the selected frontend's backend catalog supports the model architectures and `Flux2KleinPipeline`. Allow sufficient disk, host memory, and GPU memory for large presets. GPT-OSS 120B alone downloads about 63.39 GB of weights, before build layers and caches. The workflow currently uses `ubuntu-latest-16-cores`; arrange adequate runner capacity before scheduling models that exceed it.
4. After validation, rerun `update-models` with `staging: false` for the same model and runtime selection. Production images are signed and their signatures verified by the workflow. Confirm the resulting tags and digests before directing users to them.
5. Remove the pending-publication notices from the README, quickstart, and pre-made model documentation once all advertised new images are available. Validate replacements before moving deployments off retired tags. Existing registry tags are retained; removing a preset from the catalog stops future rebuilds and weekly patches.

The new Qwen, Gemma 4, Devstral Small 2, and corrected GPT-OSS recipes pin their GGUF downloads by revision and SHA-256. FLUX.2 downloads the official Hugging Face pipeline on first inference with the current Diffusers backend; that download is not revision-pinned. Include first-use download and cache behavior in staging validation.

The weekly patch workflow reads canonical production tags in `ghcr.io/kaito-project/aikit/` from the same catalog. Its existing scope excludes alias tags and Apple Silicon repositories; refresh those variants with a full `update-models` rebuild. Publish newly added tags before the next scheduled patch run. Weekly patching does not publish new presets or validate model inference. The catalog refresh alone does not establish runtime compatibility.
