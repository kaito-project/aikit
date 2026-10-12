FROM --platform=$BUILDPLATFORM golang:1.27-bookworm@sha256:5cf287a799e6b94384bad13d16b14904c531f51ba65792237e122ce42b392f61 AS builder

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
