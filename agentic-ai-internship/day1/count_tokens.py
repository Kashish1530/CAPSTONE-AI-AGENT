import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

INPUT_PRICE_PER_MILLION = 1.00
OUTPUT_PRICE_PER_MILLION = 5.00

total_input_tokens = 0

for i in range(1, 6):
    prompt = input(f"Enter prompt {i}: ")

    tokens = encoding.encode(prompt)
    token_count = len(tokens)

    total_input_tokens += token_count

    cost = (token_count / 1_000_000) * INPUT_PRICE_PER_MILLION

    print(f"Prompt {i}: {token_count} tokens | ~${cost:.6f}")

total_cost = (total_input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION

print(f"\nTotal input tokens: {total_input_tokens}")
print(f"Total estimated input cost: ${total_cost:.6f}")