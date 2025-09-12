package main

import (
	"os"

	"github.com/kaito-project/aikit/pkg/build"
	"github.com/moby/buildkit/frontend/gateway/grpcclient"
	"github.com/moby/buildkit/util/appcontext"
	"github.com/moby/buildkit/util/bklog"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc/grpclog"
)

func main() {
	bklog.L.Logger.SetOutput(os.Stderr)
	grpclog.SetLoggerV2(grpclog.NewLoggerV2WithVerbosity(bklog.L.WriterLevel(logrus.InfoLevel), bklog.L.WriterLevel(logrus.WarnLevel), bklog.L.WriterLevel(logrus.ErrorLevel), 1))

	ctx := appcontext.Context()

	// Always run through the primary build router
	if err := grpcclient.RunFromEnvironment(ctx, build.Build); err != nil {
		bklog.L.WithError(err).Fatal("error running frontend")
		os.Exit(1)
	}
}
