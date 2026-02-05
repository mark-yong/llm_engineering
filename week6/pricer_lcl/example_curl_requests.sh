#!/bin/bash
# Example curl requests for /v1/chat/completions with include_reasoning parameter
# Server should be running at http://192.168.68.104:8000

BASE_URL="http://192.168.68.104:8000"

echo "=== Example 1: Request WITH reasoning (default) ==="
curl -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France? Explain your reasoning."}
    ]
  }'

echo -e "\n\n=== Example 2: Request WITHOUT reasoning (include_reasoning: false) ==="
curl -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "include_reasoning": false
  }'

echo -e "\n\n=== Example 3: Product description (no reasoning for cleaner output) ==="
curl -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {
        "role": "system",
        "content": "Create a concise description of a product. Respond only in this format. Do not include part numbers.\nTitle: Rewritten short precise title\nCategory: eg Electronics\nBrand: Brand name\nDescription: 1 sentence description\nDetails: 1 sentence on features"
      },
      {
        "role": "user",
        "content": "Schlage F59 AND 613 Andover Interior Knob with Deadbolt, Oil Rubbed Bronze"
      }
    ],
    "include_reasoning": false
  }'

echo -e "\n"
