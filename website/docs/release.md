---
title: Release Process
---

AIKit uses protected, immutable Git tags as production release events. The supported process is:

1. Run [Prepare release](https://github.com/kaito-project/aikit/actions/workflows/prepare-release.yaml) from `main` with a stable version in `vMAJOR.MINOR.PATCH` form.
2. Open the generated pull request, wait for its checks (approving them if GitHub prompts), then review and merge it into the applicable `release-X.Y` branch.
3. Run [Publish release](https://github.com/kaito-project/aikit/actions/workflows/publish-release.yaml) from `main` with the same version. Review the validated version, branch, commit, and preparation pull request, then approve the `prod` deployment.
4. The release GitHub App creates the protected tag. The tag starts the [artifact](https://github.com/kaito-project/aikit/actions/workflows/release.yaml) and [runner-image](https://github.com/kaito-project/aikit/actions/workflows/release-runners.yaml) publishing workflows, which publish immutable versioned artifacts only.
5. After a publisher completes, the trusted [latest reconciler](https://github.com/kaito-project/aikit/actions/workflows/reconcile-release-latest.yaml) selects the highest stable version whose applicable workflow succeeded and updates its mutable image aliases. The app reconciler also assigns GitHub Latest.
6. When the released major/minor line is newer than `main`, review and merge the version-sync pull request created by the trusted publish workflow. This also covers recovery releases whose first usable tag is a patch such as `v0.22.1`.

The publish preflight requires all version files to match, the selected commit to be reachable from `release-X.Y`, and a checked, merged preparation pull request that changed both version manifests to be an ancestor. Follow-up release fixes after the preparation pull request are allowed, but both the preparation pull request and the exact selected commit must have successful lint and unit-test runs. Push CI runs even for documentation-only follow-up commits so the selected commit always has evidence.

Before publishing a new tag from a release branch created before these guardrails, backport the complete `.github/workflows` directory and the release control scripts. **Publish release** compares the complete workflow tree and host-executed release guardrails with the immutable `main` workflow revision being approved and rejects a stale release branch. This is required because tag-push workflows execute from the tagged commit.

Patch versions must increase within a release line. An older supported line can still receive a maintenance release, but it does not replace mutable aliases from a newer successful publication. App and runner `latest` aliases may remain on different versions if only one publishing workflow succeeds. If any release branch with an existing stable tag is missing, restore it from its historical ancestry before preparing the release; the workflow refuses to recreate a previously released line from `main`. A restored branch must descend from the latest stable tag on that line.

Never create, push, force-update, or delete a `v*` tag manually. A tag is the deployment trigger, not a preparation step. For a transient publication failure, rerun the failed workflow against the same tag. If the release commit must change, prepare a new patch version instead of moving the existing tag. If `main` advances while **Publish release** is awaiting approval, rerun it from the new `main` revision.

Rerunning **Publish release** for an existing immutable tag is only for recovery work such as version synchronization. It does not recreate the tag or retrigger the artifact and runner-image workflows. Rerun a failed publishing workflow directly only for a tag created through this protected flow; review the provenance of legacy tags first.

If a mutable alias or GitHub Latest needs repair after a transient failure, run **Reconcile release latest** from `main`. It recomputes the target from current immutable tags and successful publisher runs; it never trusts the triggering run's version.

After publishing finishes, run [Update models](https://github.com/kaito-project/aikit/actions/workflows/update-models.yaml) to refresh the pre-built models.

Repository administrators can find the required GitHub App, environment, and tag-ruleset configuration in the [contributor release setup](https://github.com/kaito-project/aikit/blob/main/CONTRIBUTING.md#release-repository-setup).
