VERSION := v0.21.0

REGISTRY ?= ghcr.io/kaito-project
REPOSITORY ?= /aikit
KIND_VERSION ?= 0.29.0
KUBERNETES_VERSION ?= 1.33.2
HELM_VERSION ?= 3.18.3
TAG ?= test
OUTPUT_TYPE ?= type=docker
TEST_IMAGE_NAME ?= testmodel
TEST_FILE ?= test/aikitfile-llama.yaml
RUNTIME ?= ""
PLATFORMS ?= linux/amd64,linux/arm64
GOLANGCI_LINT_VERSION ?= v2.11.2

GIT_COMMIT := $(shell git rev-list --abbrev-commit --tags --max-count=1)
GIT_TAG := $(shell git describe --abbrev=0 --tags ${GIT_COMMIT} 2>/dev/null || true)
LDFLAGS := "-X github.com/kaito-project/aikit/pkg/version.Version=$(GIT_TAG:%=%)"

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: lint
lint: ## Run golangci-lint over the module
	go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION) run -v ./... --timeout 5m

.PHONY: build-aikit
build-aikit: ## Build the AIKit frontend image
	docker buildx build . -t ${REGISTRY}${REPOSITORY}/aikit:${TAG} \
		--output=${OUTPUT_TYPE} \
		--build-arg LDFLAGS=${LDFLAGS} \
		--platform ${PLATFORMS} \
		--progress=plain

.PHONY: build-test-model
build-test-model: ## Build a test model image from TEST_FILE
	docker buildx build . -t ${REGISTRY}${REPOSITORY}/${TEST_IMAGE_NAME}:${TAG} -f ${TEST_FILE} \
		--progress=plain --provenance=false \
		--output=${OUTPUT_TYPE} \
		--build-arg runtime=${RUNTIME} \
		--platform ${PLATFORMS}

.PHONY: build-base
build-base: ## Build and push the base image
	docker buildx build . -t ${REGISTRY}${REPOSITORY}/base:latest -f Dockerfile.base \
		--platform ${PLATFORMS} \
		--output=${OUTPUT_TYPE} \
		--sbom=true --push

.PHONY: run-test-model
run-test-model: ## Run the test model image
	docker run --rm -p 8080:8080 ${REGISTRY}${REPOSITORY}/${TEST_IMAGE_NAME}:${TAG}

.PHONY: run-test-model-gpu
run-test-model-gpu: ## Run the test model image with GPU
	docker run --rm -p 8080:8080 --gpus all ${REGISTRY}${REPOSITORY}/${TEST_IMAGE_NAME}:${TAG}

.PHONY: run-test-model-rocm
run-test-model-rocm: ## Run the test model image with ROCm
	docker run --rm -p 8080:8080 --device /dev/kfd --device /dev/dri --group-add video --group-add $$(stat -c '%g' /dev/dri/renderD128) \
		${REGISTRY}${REPOSITORY}/${TEST_IMAGE_NAME}:${TAG}

.PHONY: run-test-model-applesilicon
run-test-model-applesilicon: ## Run the test model image on Apple Silicon
	podman run --rm -p 8080:8080 --device /dev/dri ${REGISTRY}${REPOSITORY}/${TEST_IMAGE_NAME}:${TAG}

.PHONY: test
test: ## Run unit tests with race detector and coverage
	go test -v ./... -race -coverprofile=coverage.txt -covermode=atomic

.PHONY: test-e2e-dependencies
test-e2e-dependencies: ## Install e2e test dependencies (kubectl, helm, kind)
	mkdir -p ${GITHUB_WORKSPACE}/bin
	echo "${GITHUB_WORKSPACE}/bin" >> ${GITHUB_PATH}

	# used for kubernetes test
	curl -sSL https://dl.k8s.io/release/v${KUBERNETES_VERSION}/bin/linux/amd64/kubectl -o ${GITHUB_WORKSPACE}/bin/kubectl && chmod +x ${GITHUB_WORKSPACE}/bin/kubectl
	curl https://get.helm.sh/helm-v${HELM_VERSION}-linux-amd64.tar.gz | tar xz && mv linux-amd64/helm ${GITHUB_WORKSPACE}/bin/helm && chmod +x ${GITHUB_WORKSPACE}/bin/helm
	curl -sSL https://github.com/kubernetes-sigs/kind/releases/download/v${KIND_VERSION}/kind-linux-amd64 -o ${GITHUB_WORKSPACE}/bin/kind && chmod +x ${GITHUB_WORKSPACE}/bin/kind

.PHONY: release-manifest
release-manifest: ## Bump chart and Makefile versions to NEWVERSION
	@sed -i "s/appVersion: $(VERSION)/appVersion: ${NEWVERSION}/" ./charts/aikit/Chart.yaml
	@sed -i "s/version: $$(echo ${VERSION} | cut -c2-)/version: $$(echo ${NEWVERSION} | cut -c2-)/" ./charts/aikit/Chart.yaml
	@sed -i -e 's/^VERSION := $(VERSION)/VERSION := ${NEWVERSION}/' ./Makefile
