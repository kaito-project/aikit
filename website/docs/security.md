---
title: Supply Chain Security
---

AIKit is designed with security in mind. Our approach to supply chain security includes detailed tracking of software components, transparent build processes, and proactive vulnerability management. This ensures that every part of our software ecosystem remains secure and trustworthy.

## SBOM (Software Bill of Materials)

AIKit publishes [Software Bill of Materials (SBOM)](https://www.cisa.gov/sbom) for each release and for all [pre-made models](premade-models.md). The SBOM is a comprehensive list of all the components and dependencies used in the project, detailing their versions, licenses, and sources. This transparency helps users and stakeholders understand what software is included, facilitating better risk management and compliance with security and licensing requirements.

To access the SBOM for a specific AIKit image, use the following command:

```bash
# update this with the image you want to inspect
IMAGE=ghcr.io/kaito-project/aikit/llama3.1:8b
docker buildx imagetools inspect $IMAGE --format "{{ json .SBOM.SPDX }}"
```

The output will provide a detailed JSON document listing all the software components in the image, including direct and transitive dependencies. For more information, please visit [Docker SBOM documentation](https://docs.docker.com/build/attestations/sbom/).

## Provenance attestation

Provenance attestation provides a detailed record of how and where an image was built, offering transparency and trust in the build process. AIKit uses BuildKit to generate and publish provenance data for each of its images. This data includes information about the build environment, the build process, and the source control context, ensuring that the images are traceable and verifiable from their origins to their final state.

To inspect the provenance attestation for an AIKit image, you can use the following command:

```bash
# update this with the image you want to inspect
IMAGE=ghcr.io/kaito-project/aikit/llama3.1:8b
docker buildx imagetools inspect $IMAGE --format "{{ json .Provenance.SLSA }}"
```

This command will output a JSON file containing the build provenance details, including the source repository, commit hash, build configuration, and more. This helps verify that the image was built from trusted sources and has not been tampered with. For more information, please visit [Docker Provenance documentation](https://docs.docker.com/build/attestations/slsa-provenance/).

## Vulnerability Patching

Ensuring that our images are free from known vulnerabilities is crucial. Not only AIKit uses a custom distroless-based base image to reduce the number of vulnerabilities, attack surface and size, AIKit uses [Copacetic](https://github.com/project-copacetic/copacetic) to scan and patch OS-based vulnerabilities for all [pre-made models](premade-models.md) on a weekly basis. Copacetic automates the process of identifying and remediating security issues, helping us maintain a robust and secure software supply chain.

Every week, Copacetic performs the following actions:

- Scan: It analyzes the images for vulnerabilities using [Trivy](https://github.com/aquasecurity/trivy) against a comprehensive database of known security issues.
- Patch: It automatically applies patches or updates to mitigate any identified vulnerabilities using [Copacetic](https://github.com/project-copacetic/copacetic).
- Publish: It updates the images with the latest security fixes and publishes them to our container registry.

This automated and regular process ensures that our users always have access to the most secure and up-to-date images. You can monitor the status and results of these scans on our security dashboard.

## Image Signature Verification

AIKit and pre-made models are keyless signed with OIDC in GitHub Actions with [cosign](https://github.com/sigstore/cosign). You can verify the images with the following commands:

```bash
IMAGE=ghcr.io/kaito-project/aikit/llama2:7b # update this with the image you want to verify
DIGEST=$(cosign triangulate ${IMAGE} --type digest)
cosign verify ${DIGEST} \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity-regexp 'https://github\.com/kaito-project/aikit/\.github/workflows/.+'
```

You should see an output similar to the following:

```bash
Verification for ghcr.io/kaito-project/aikit/llama2@sha256:d47fdba491a9a47ce4911539a77e0c0a12b2e14f5beed88cb8072924b02130b4 --
The following checks were performed on each of these signatures:
  - The cosign claims were validated
  - Existence of the claims in the transparency log was verified offline
  - The code-signing certificate was verified using trusted certificate authority certificates
...
```
