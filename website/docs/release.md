---
title: Release Process
---

AIKit uses protected, immutable Git tags as production release events. The supported process is:

1. Run [Prepare release](https://github.com/kaito-project/aikit/actions/workflows/prepare-release.yaml) from `main` with a stable version in `vMAJOR.MINOR.PATCH` form.
2. Review and merge the generated pull request into the applicable `release-X.Y` branch.
3. Run [Publish release](https://github.com/kaito-project/aikit/actions/workflows/publish-release.yaml) from `main` with the same version. Review the validated version, branch, commit, and preparation pull request, then approve the `prod` deployment.
4. The release GitHub App creates the protected tag. The tag starts the [artifact](https://github.com/kaito-project/aikit/actions/workflows/release.yaml) and [runner-image](https://github.com/kaito-project/aikit/actions/workflows/release-runners.yaml) publishing workflows.
5. For a new minor version, review and merge the generated version-sync pull request to `main`.

The publish preflight requires all version files to match, the selected commit to be reachable from `release-X.Y`, and the merged preparation pull request to be an ancestor. Follow-up release fixes after the preparation pull request are allowed.

Never create, push, force-update, or delete a `v*` tag manually. A tag is the deployment trigger, not a preparation step. For a transient publication failure, rerun the failed workflow against the same tag. If the release commit must change, prepare a new patch version instead of moving the existing tag.

After publishing finishes, run [Update models](https://github.com/kaito-project/aikit/actions/workflows/update-models.yaml) to refresh the pre-built models.

Repository administrators can find the required GitHub App, environment, and tag-ruleset configuration in the [contributor release setup](https://github.com/kaito-project/aikit/blob/main/CONTRIBUTING.md#release-repository-setup).
