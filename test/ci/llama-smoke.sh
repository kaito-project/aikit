#!/usr/bin/env bash
set -euo pipefail

model_name="${1:?Usage: llama-smoke.sh MODEL [chat|tools]}"
test_mode="${2:-chat}"

case "$test_mode" in
  chat)
    token_limit=64
    payload=$(jq -n --arg model "$model_name" --argjson max_tokens "$token_limit" '{
      model: $model,
      messages: [{role: "user", content: "What is the capital of France? Reply with only the city name."}],
      max_tokens: $max_tokens,
      temperature: 0
    }')
    ;;
  tools)
    token_limit=128
    payload=$(jq -n --arg model "$model_name" --argjson max_tokens "$token_limit" '{
      model: $model,
      messages: [{role: "user", content: "What is the weather in Paris?"}],
      max_tokens: $max_tokens,
      temperature: 0,
      tools: [{
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the current weather for a location",
          parameters: {
            type: "object",
            properties: {location: {type: "string", description: "The city name"}},
            required: ["location"]
          }
        }
      }]
    }')
    ;;
  *)
    printf 'Unknown test mode: %s\n' "$test_mode" >&2
    exit 2
    ;;
esac

response=$(curl --fail --silent --show-error --max-time 120 \
  --retry 60 --retry-all-errors --retry-delay 1 --retry-max-time 180 \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' --data "$payload")
printf '%s\n' "$response"

printf '%s\n' "$response" | jq -e \
  --arg model "$model_name" --arg mode "$test_mode" --argjson max_tokens "$token_limit" '
  .error == null and
  .object == "chat.completion" and
  .model == $model and
  (.choices | length == 1) and
  .choices[0].message.role == "assistant" and
  (.usage.completion_tokens > 0 and .usage.completion_tokens <= $max_tokens) and
  (if $mode == "chat" then
    .choices[0].finish_reason == "stop" and
    (.choices[0].message.content | test("(^|[^[:alpha:]])Paris([^[:alpha:]]|$)"; "i"))
  else
    .choices[0].finish_reason == "tool_calls" and
    (.choices[0].message.tool_calls | length == 1) and
    .choices[0].message.tool_calls[0].type == "function" and
    .choices[0].message.tool_calls[0].function.name == "get_weather" and
    (.choices[0].message.tool_calls[0].function.arguments | fromjson | .location |
      test("(^|[^[:alpha:]])Paris([^[:alpha:]]|$)"; "i"))
  end)'
