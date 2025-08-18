FROM golang:1.25-bookworm@sha256:74908ad827a5849c557eeca81d46263acf788ead606102d83466f499f83e35b1 AS builder
ARG LDFLAGS
COPY . /go/src/github.com/kaito-project/aikit
WORKDIR /go/src/github.com/kaito-project/aikit
RUN CGO_ENABLED=0 go build -o /aikit -ldflags "${LDFLAGS} -w -s -extldflags '-static'" ./cmd/frontend

FROM scratch
COPY --from=builder /aikit /bin/aikit
ENTRYPOINT ["/bin/aikit"]
