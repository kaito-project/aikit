FROM --platform=$BUILDPLATFORM golang:1.25-bookworm@sha256:81dc45d05a7444ead8c92a389621fafabc8e40f8fd1a19d7e5df14e61e98bc1a AS builder

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT=""
ARG LDFLAGS

COPY . /go/src/github.com/kaito-project/aikit
WORKDIR /go/src/github.com/kaito-project/aikit
RUN CGO_ENABLED=0 \
    GOOS=${TARGETOS} \
    GOARCH=${TARGETARCH} \
    GOARM=${TARGETVARIANT} \
    go build -o /aikit -ldflags "${LDFLAGS} -w -s -extldflags '-static'" ./cmd/frontend

FROM scratch
LABEL org.opencontainers.image.source="https://github.com/kaito-project/aikit"
# Mark this image as a BuildKit gateway frontend so it can be used via #syntax or --syntax
LABEL moby.buildkit.frontend="gateway.v0"
# Back-compat with older tooling that looks for the legacy label key
LABEL org.mobyproject.buildkit.frontend="gateway.v0"
# Provide system CA certificates so HTTPS (Hugging Face, etc.) works inside the frontend
COPY --from=builder /etc/ssl/certs /etc/ssl/certs
COPY --from=builder /aikit /bin/aikit
ENTRYPOINT ["/bin/aikit"]
