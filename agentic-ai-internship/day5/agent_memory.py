import ollama
import json
import os

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_memory(messages):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(messages, f, indent=2)

conversation = load_memory()

print("=== Agent with Memory ===")
print("(Type 'quit' to exit. Your conversation will be remembered next time.)\n")

if conversation:
    print(f"[Loaded {len(conversation)} previous messages from memory]\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    conversation.append({'role': 'user', 'content': user_input})

    response = ollama.chat(model='llama3.1', messages=conversation)
    reply = response['message']['content']

    print("Agent:", reply, "\n")

    conversation.append({'role': 'assistant', 'content': reply})

    save_memory(conversation)

print("Conversation saved. Goodbye!")

