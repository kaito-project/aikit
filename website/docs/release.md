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

[`models/catalog.json`](https://github.com/kaito-project/aikit/blob/main/models/catalog.json) lists the active presets used by [publishing](https://github.com/kaito-project/aikit/actions/workflows/update-models.yaml) and [weekly patching](https://github.com/kaito-project/aikit/actions/workflows/patch-models.yaml).

After releasing the frontend, run `update-models` with `staging: true`. Set `models` to a JSON array of preset IDs, or `[]` for all active presets. Validate the staged images, then rerun with `staging: false` and the same models and runtimes to publish.

Weekly patching updates canonical tags in `ghcr.io/kaito-project/aikit/`; alias tags and Apple Silicon images require a full `update-models` rebuild.
