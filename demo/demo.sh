#!/bin/bash

. third_party/demo-magic/demo-magic.sh

clear
export DEMO_PROMPT="${GREEN}➜  ${COLOR_RESET}"

echo "✨ In this demo, we are going to start by fine tuning a model and then deploy the model as a minimal container!"

echo ""

echo "👷‍ First, we are going to create a new builder"

echo ""

pei "docker buildx create --name aikit-builder --use --buildkitd-flags '--allow-insecure-entitlement security.insecure'"

echo ""

echo "🗃️ Create a configuration for the fine tuning. We are going to be using a Mistral model and fine tune using OpenHermes dataset."

cat > aikit-finetune.yaml << EOF
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
baseModel: unsloth/mistral-7b-instruct-v0.2-bnb-4bit
datasets:
  - source: "teknium/openhermes"
    type: "alpaca"
config:
  unsloth:
EOF

echo ""

pei "bat aikit-finetune.yaml"

echo ""

echo "🎵 Starting the fine tuning process using the above configuration file, and output fine tuned model will be saved in _output folder."

echo ""

pei "docker buildx build --allow security.insecure --file 'aikit-finetune.yaml' --output '_output' --target unsloth --progress plain ."

echo ""

echo "✅ We have finished fine tuning the model. Let's look at the output..."

echo ""

pei "ls -al _output"

echo ""

echo "📦 Now that we have a fine tuned model. We can deploy it as a minimal container."

echo ""

echo "📃 We'll start by creating a basic inference configuration file for the deployment."

cat > aikit-inference.yaml << EOF
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
debug: true
apiVersion: v1alpha1
runtime: cuda
models:
  - name: mistral-finetuned
    source: aikit-model-q4_k_m.gguf
    promptTemplates:
      - name: instruct
        template: |
          Below is an instruction that describes a task. Write a response that appropriately completes the request. Keep your responses concise.

          ### Instruction:
          {{.Input}}

          ### Response:
config: |
  - name: mistral-finetuned
    parameters:
      model: aikit-model-q4_k_m.gguf
    context_size: 4096
    gpu_layers: 35
    f16: true
    mmap: true
    template:
      chat: instruct
EOF

pei "bat aikit-inference.yaml"

echo ""

echo "🏗️ We can now build a minimal container for the model using the configuration file."

echo ""

pei "docker buildx build -t mistral-finetuned -f aikit-inference.yaml --load --progress plain _output"

echo ""

echo "🏃 We have finished building the minimal container. Let's start the container and test it."

echo ""

pei "docker run --name mistral-finetuned -d --rm -p 8080:8080 --gpus all mistral-finetuned"

echo ""

echo "🧪 We can now test the container using a sample query. Since this is OpenAI API compatible, you can use it as a drop-in replacement for any application that uses OpenAI API."

echo ""

pei "curl http://localhost:8080/v1/chat/completions -H \"Content-Type: application/json\" -d '{\"model\": \"mistral-finetuned\", \"messages\": [{\"role\": \"user\", \"content\": \"Generate a list of 10 words that start with ab\"}]}'"

echo ""

echo "🙌 We have successfully deployed the fine tuned model as a minimal container and successfully verified it! We can now stop the container if we wish."

echo ""

pei "docker stop mistral-finetuned"

echo ""

echo "❤️ In this demo, we have shown how to fine tune a model and deploy it as a minimal container using AIKit. Thank you for watching!"

echo ""

# pei "docker buildx rm aikit-builder"
