import ollama
import sqlite3
from datetime import datetime

def query_database(sql):
    conn = sqlite3.connect("demo.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS employees (name TEXT, department TEXT)")
    cursor.execute("DELETE FROM employees")
    cursor.executemany("INSERT INTO employees VALUES (?, ?)", [
        ("Rohan", "Engineering"),
        ("Priya", "Marketing"),
    ])
    conn.commit()
    result = cursor.execute(sql).fetchall()
    conn.close()
    return str(result)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def send_email(to, subject, body):
    return f"Email sent to {to}: '{subject}' — {body}"

tools = [
    {'type': 'function', 'function': {'name': 'query_database', 'description': 'Query employees database (columns: name, department)',
        'parameters': {'type': 'object', 'properties': {'sql': {'type': 'string'}}, 'required': ['sql']}}},
    {'type': 'function', 'function': {'name': 'get_current_time', 'description': 'Get current date and time',
        'parameters': {'type': 'object', 'properties': {}}}},
    {'type': 'function', 'function': {'name': 'send_email', 'description': 'Send an email',
        'parameters': {'type': 'object', 'properties': {'to': {'type': 'string'}, 'subject': {'type': 'string'}, 'body': {'type': 'string'}}, 'required': ['to', 'subject', 'body']}}},
]

def run_tool(name, args):
    if name == 'query_database':
        return query_database(args['sql'])
    elif name == 'get_current_time':
        return get_current_time()
    elif name == 'send_email':
        return send_email(args['to'], args['subject'], args['body'])
    return "Unknown tool"

messages = [{'role': 'user', 'content': 
    "Find who works in Engineering, check the current time, then send them an email reminding them of a meeting at that time."}]

print("STARTING TASK...\n")

for step in range(5): 
    response = ollama.chat(model='llama3.1', messages=messages, tools=tools)
    messages.append(response['message'])

    if response['message'].get('tool_calls'):
        for call in response['message']['tool_calls']:
            name = call['function']['name']
            args = call['function']['arguments']
            result = run_tool(name, args)

            print(f"STEP {step+1}: Called '{name}' with {args}")
            print(f"  -> Result: {result}\n")

            messages.append({'role': 'tool', 'content': result})
    else:
        print("FINAL ANSWER:", response['message']['content'])
        break
