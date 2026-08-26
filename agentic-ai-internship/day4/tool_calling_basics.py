import ollama
from datetime import datetime

# Step 1: Define real Python functions (the "tools")
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Step 2: Describe these tools to the model
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'calculator',
            'description': 'Evaluate a math expression, e.g. "2 + 2" or "15 * 3"',
            'parameters': {
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string', 'description': 'The math expression to evaluate'}
                },
                'required': ['expression']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Get the current date and time',
            'parameters': {'type': 'object', 'properties': {}}
        }
    }
]

def run_tool_call(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': 'what is the capital of japan'}],
        tools=tools
    )

    if response['message'].get('tool_calls'):
        for call in response['message']['tool_calls']:
            func_name = call['function']['name']
            args = call['function']['arguments']

            print(f"Model chose tool: {func_name}")
            print(f"Arguments: {args}")

            if func_name == 'calculator':
                result = calculator(args['expression'])
            elif func_name == 'get_current_time':
                result = get_current_time()
            else:
                result = "Unknown tool"

            print(f"Tool result: {result}")
    else:
        print("Model did not call a tool. Raw answer:", response['message']['content'])

# Test 1: Should trigger the calculator
run_tool_call("What is 47 multiplied by 89?")

# Test 2: Should trigger get_current_time
run_tool_call("What time is it right now?")

# Test 3: Should NOT trigger any tool (general knowledge)
run_tool_call("What is the capital of Japan?")


