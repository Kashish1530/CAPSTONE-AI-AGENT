import ollama
prompt='WRITE A ONE SENTENCE TAGLINE FOR A Beauty product'
temperatures=[0.0, 0.7, 1.2]

for temp in temperatures:
    response = ollama.generate(
        model='llama3.1',
        prompt=prompt,
        options={'temperature': float(temp)}
)
    print(f"\n--- Temperature: {temp}---")
    print(response['response'])
