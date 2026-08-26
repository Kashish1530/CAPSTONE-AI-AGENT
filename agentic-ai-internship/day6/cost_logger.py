import ollama
import time
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

# Pricing using Claude Haiku 4.5 rates as reference, since Ollama for free
INPUT_PRICE = 1.00 / 1_000_000
OUTPUT_PRICE = 5.00 / 1_000_000

def timed_call(prompt, model='llama3.1'):
    start = time.time()
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    elapsed = time.time() - start

    input_tokens = len(encoding.encode(prompt))
    output_tokens = len(encoding.encode(response['message']['content']))
    cost = (input_tokens * INPUT_PRICE) + (output_tokens * OUTPUT_PRICE)

    return {
        "answer": response['message']['content'],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "latency": elapsed
    }

questions = [
    "What is the capital of France?",
    "Explain photosynthesis in detail, covering the light and dark reactions, the role of chlorophyll, and how energy from sunlight is converted into chemical energy stored in glucose molecules.",
    "What is 15 + 27?",
]

print("=== BEFORE OPTIMIZATION (verbose prompts) ===\n")
total_cost_before = 0
for q in questions:
    result = timed_call(q)
    total_cost_before += result["cost"]
    print(f"Q: {q[:60]}...")
    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out | Cost: ${result['cost']:.6f} | Latency: {result['latency']:.2f}s\n")

print(f"TOTAL COST (before): ${total_cost_before:.6f}\n")

print("=== AFTER OPTIMIZATION (concise instruction added) ===\n")
total_cost_after = 0
for q in questions:
    concise_prompt = q + " Answer in 1-2 sentences maximum."
    result = timed_call(concise_prompt)
    total_cost_after += result["cost"]
    print(f"Q: {q[:60]}...")
    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out | Cost: ${result['cost']:.6f} | Latency: {result['latency']:.2f}s\n")

print(f"TOTAL COST (after): ${total_cost_after:.6f}")

savings = ((total_cost_before - total_cost_after) / total_cost_before) * 100
print(f"\nCOST REDUCTION: {savings:.1f}%")