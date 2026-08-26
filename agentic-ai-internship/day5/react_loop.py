import ollama

def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def word_count(text):
    return str(len(text.split()))

tools = [
    {'type': 'function', 'function': {'name': 'calculator', 'description': 'Evaluate a math expression',
        'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}},
    {'type': 'function', 'function': {'name': 'word_count', 'description': 'Count words in a piece of text',
        'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}},
]

def run_tool(name, args):
    if name == 'calculator':
        return calculator(args['expression'])
    elif name == 'word_count':
        return word_count(args['text'])
    return "Unknown tool"

task = "What is 25 times 8, and then how many words are in the sentence 'The quick brown fox jumps over the lazy dog'?"

messages = [{'role': 'user', 'content': task}]

print(f"TASK: {task}\n")

max_steps = 6
for step in range(max_steps):
    print(f"--- Step {step + 1} ---")

    response = ollama.chat(model='llama3.1', messages=messages, tools=tools)
    messages.append(response['message'])

    if response['message'].get('tool_calls'):
        for call in response['message']['tool_calls']:
            name = call['function']['name']
            args = call['function']['arguments']

            print(f"THOUGHT: I need to use '{name}'")
            print(f"ACTION: {name}({args})")

            result = run_tool(name, args)
            print(f"OBSERVATION: {result}\n")

            messages.append({'role': 'tool', 'content': result})
    else:
        print("FINAL ANSWER:", response['message']['content'])
        break
else:
    print("Stopped: reached max steps without finishing.")
