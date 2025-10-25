package utils // nolint:revive

import (
	"testing"
)

func Test_FileNameFromURL(t *testing.T) {
	type args struct {
		urlString string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "simple",
			args: args{urlString: "http://foo.bar/baz"},
			want: "baz",
		},
		{
			name: "complex",
			args: args{urlString: "http://foo.bar/baz.tar.gz"},
			want: "baz.tar.gz",
		},
		{
			name: "complex with path",
			args: args{urlString: "http://foo.bar/baz.tar.gz?foo=bar"},
			want: "baz.tar.gz",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FileNameFromURL(tt.args.urlString); got != tt.want {
				t.Errorf("FileNameFromURL() = %v, want %v", got, tt.want)
			}
		})
	}
}
