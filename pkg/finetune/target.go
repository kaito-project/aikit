package finetune

import _ "embed"

// TargetUnsloth contains the Unsloth fine-tuning entrypoint used by the BuildKit frontend.
//
//go:embed target_unsloth.py
var TargetUnsloth []byte

// UnslothPylock contains the fully resolved Python environment for CUDA 12.6 and Python 3.10.
//
//go:embed pylock.toml
var UnslothPylock []byte

// UVBootstrap contains the hashed uv installer requirement used to create the isolated environment.
//
//go:embed uv-bootstrap.txt
var UVBootstrap []byte
