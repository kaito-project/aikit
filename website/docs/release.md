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

- Once release is done, trigger [update models](https://github.com/kaito-project/aikit/actions/workflows/update-models.yaml) action to update the pre-built models.

- Mixtral 8x7b and Llama 3 70b models does not fit into GitHub runners due to their size. Trigger self-hosted [update models](https://github.com/kaito-project/aikit/actions/workflows/update-models-self.yaml) action to update these pre-built models.
