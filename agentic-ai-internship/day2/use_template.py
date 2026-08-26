import ollama
with open("invoice_extractv1.txt", "r") as f:
    template = f.read()
invoice_text = "Aditya Bhatt, owner of Skyline Traders, invoiced $99.00 on 2026-01-25"

prompt = template.replace("{text}", invoice_text)
response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])

