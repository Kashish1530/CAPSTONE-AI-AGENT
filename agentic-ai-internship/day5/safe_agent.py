import ollama
import time

def delete_file(filename):
    return f"[SIMULATED] File '{filename}' would be deleted."

def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    {'type': 'function', 'function': {'name': 'delete_file', 'description': 'Delete a file (destructive action)',
        'parameters': {'type': 'object', 'properties': {'filename': {'type': 'string'}}, 'required': ['filename']}}},
    {'type': 'function', 'function': {'name': 'calculator', 'description': 'Evaluate a math expression',
        'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}},
]

DESTRUCTIVE_TOOLS = {'delete_file'}

def run_tool(name, args):
    if name == 'delete_file':
        return delete_file(args['filename'])
    elif name == 'calculator':
        return calculator(args['expression'])
    return "Unknown tool"

MAX_STEPS = 5

MAX_SECONDS = 30
start_time = time.time()

task = "Calculate 12 times 7, then delete the file called old_report.txt"
messages = [{'role': 'user', 'content': task}]

print(f"TASK: {task}\n")

for step in range(MAX_STEPS):
    elapsed = time.time() - start_time
    if elapsed > MAX_SECONDS:
        print(f"STOPPED: Exceeded {MAX_SECONDS} second timeout.")
        break

    print(f"--- Step {step + 1} (elapsed: {elapsed:.1f}s) ---")

    response = ollama.chat(model='llama3.1', messages=messages, tools=tools)
    messages.append(response['message'])

    if response['message'].get('tool_calls'):
        for call in response['message']['tool_calls']:
            name = call['function']['name']
            args = call['function']['arguments']

            if name in DESTRUCTIVE_TOOLS:
                print(f"  Agent wants to run a DESTRUCTIVE action: {name}({args})")
                approval = input("Approve this action? (yes/no): ")
                if approval.lower() != 'yes':
                    result = "Action denied by user."
                    print("Action denied.\n")
                    messages.append({'role': 'tool', 'content': result})
                    continue

            result = run_tool(name, args)
            print(f"Action: {name}({args}) -> {result}\n")
            messages.append({'role': 'tool', 'content': result})
    else:
        print("FINAL ANSWER:", response['message']['content'])
        break
else:
    print(f"STOPPED: Reached max steps ({MAX_STEPS}) without finishing.")