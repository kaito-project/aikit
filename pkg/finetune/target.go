package finetune

import _ "embed"

// TargetUnsloth contains the Unsloth fine-tuning entrypoint used by the BuildKit frontend.
//
//go:embed target_unsloth.py
var TargetUnsloth []byte
