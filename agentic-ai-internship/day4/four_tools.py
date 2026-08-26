import ollama
import sqlite3

def web_search(query):
    if "weather" in query.lower():
        return "It is sunny and 24°C today."
    return "No results found."

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except:
        return "File not found."

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

def send_email(to, subject, body):
    return f"Email sent to {to}: {subject}"

tools = [
    {'type': 'function', 'function': {'name': 'web_search', 'description': 'Search the web',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
        'parameters': {'type': 'object', 'properties': {'filename': {'type': 'string'}}, 'required': ['filename']}}},
    {'type': 'function', 'function': {'name': 'query_database', 'description': 'Query employees database (name, department)',
        'parameters': {'type': 'object', 'properties': {'sql': {'type': 'string'}}, 'required': ['sql']}}},
    {'type': 'function', 'function': {'name': 'send_email', 'description': 'Send an email',
        'parameters': {'type': 'object', 'properties': {'to': {'type': 'string'}, 'subject': {'type': 'string'}, 'body': {'type': 'string'}}, 'required': ['to', 'subject', 'body']}}},
]

def ask(question):
    print("\nQuestion:", question)
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': question}], tools=tools)

    if response['message'].get('tool_calls'):
        call = response['message']['tool_calls'][0]
        name = call['function']['name']
        args = call['function']['arguments']
        print("Called:", name, args)

        if name == 'web_search':
            print("Result:", web_search(args['query']))
        elif name == 'read_file':
            print("Result:", read_file(args['filename']))
        elif name == 'query_database':
            print("Result:", query_database(args['sql']))
        elif name == 'send_email':
            print("Result:", send_email(args['to'], args['subject'], args['body']))
    else:
        print("No tool used:", response['message']['content'])

ask("What's the weather today?")
ask("Who is in the Engineering department?")
ask("Email priya@example.com saying the meeting is at 3pm.")




