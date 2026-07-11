FROM --platform=$BUILDPLATFORM golang:1.26-bookworm@sha256:b305420a68d0f229d91eb3b3ed9e519fcf2cf5461da4bef997bf927e8c0bfd2b AS builder

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
COPY --from=builder /etc/ssl/certs /etc/ssl/certs
COPY --from=builder /aikit /bin/aikit
ENTRYPOINT ["/bin/aikit"]
