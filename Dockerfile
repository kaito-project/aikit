FROM --platform=$BUILDPLATFORM golang:1.25-bookworm@sha256:09f53deea14d4019922334afe6258b7b776afc1d57952be2012f2c8c4076db05 AS builder

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
