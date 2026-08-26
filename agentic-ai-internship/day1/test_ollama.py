import ollama
response = ollama.chat(
    model='llama3.1',
    messages=[
        {
            'role': 'user','content': 'say hello and tell me fact about octopuses'
        }
    ]
)

print(response['message']['content'])