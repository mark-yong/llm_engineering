"""
Example of using /v1/chat/completions with include_reasoning parameter
for GPT-OSS model via vLLM server.

Reference: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#usage
"""

import json

# Example 1: Request WITH reasoning (default behavior)
# This will return both the reasoning/CoT (Chain of Thought) and the final answer
request_with_reasoning = {
    "custom_id": "example_1",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of France? Explain your reasoning."
            }
        ]
        # include_reasoning is not specified, defaults to true
        # The response will contain reasoning/CoT tokens
    }
}

# Example 2: Request WITHOUT reasoning (include_reasoning: false)
# This will skip the CoT and return only the final answer
request_without_reasoning = {
    "custom_id": "example_2",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "include_reasoning": False  # Skip Chain of Thought in output
    }
}

# Example 3: Explicitly include reasoning
request_explicit_reasoning = {
    "custom_id": "example_3",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Solve this math problem: 15 * 23 = ?"
            }
        ],
        "include_reasoning": True  # Explicitly include reasoning (default)
    }
}

# Example 4: Real-world product description example (like your use case)
# Without reasoning for cleaner output
product_request_no_reasoning = {
    "custom_id": "product_1",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "Create a concise description of a product. Respond only in this format. Do not include part numbers.\nTitle: Rewritten short precise title\nCategory: eg Electronics\nBrand: Brand name\nDescription: 1 sentence description\nDetails: 1 sentence on features"
            },
            {
                "role": "user",
                "content": "Schlage F59 AND 613 Andover Interior Knob with Deadbolt, Oil Rubbed Bronze (Interior Half Only)"
            }
        ],
        "include_reasoning": False  # Skip reasoning for cleaner, faster output
    }
}

# Print examples as JSONL format (like your input files)
examples = [
    request_with_reasoning,
    request_without_reasoning,
    request_explicit_reasoning,
    product_request_no_reasoning
]

print("Example requests (JSONL format):\n")
for example in examples:
    print(json.dumps(example, ensure_ascii=False))
    print()

# Expected response structure (when include_reasoning=False):
# The response will have the standard OpenAI format:
# {
#   "id": "...",
#   "choices": [{
#     "message": {
#       "role": "assistant",
#       "content": "Paris"  # Only final answer, no reasoning steps
#     },
#     "finish_reason": "stop"
#   }]
# }

# When include_reasoning=True (default):
# The response may contain reasoning tokens before the final answer
# {
#   "id": "...",
#   "choices": [{
#     "message": {
#       "role": "assistant",
#       "content": "[reasoning steps]... Paris"  # Includes reasoning
#     },
#     "finish_reason": "stop"
#   }]
# }
