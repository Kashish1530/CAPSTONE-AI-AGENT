
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=30)

all_chunks = []
all_metadata = []

for pdf in pdf_files:
    reader = PdfReader(pdf)
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": pdf, "page": page_num})

client = chromadb.Client()
collection = client.create_collection(name="qa_bot")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadata,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

def ask_question(question):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_chunks = results['documents'][0]
    sources = results['metadatas'][0]

    context = ""
    for chunk, meta in zip(retrieved_chunks, sources):
        context += f"[Source: {meta['source']}, Page {meta['page']}]\n{chunk}\n\n"

    prompt = f"""Answer the question using ONLY the context below. 
After your answer, list which sources you used (filename and page).

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

questions = [
    "How has private industry changed the cost of space travel?",
    "What role do healthy fats play in the body?",
    "Why is energy storage important for renewable energy?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    answer = ask_question(q)
    print(answer)

import ollama
from datetime import datetime

def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        messages=[{'role': 'user', 'content': question}],
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

run_tool_call("What is 47 multiplied by 89?")

run_tool_call("What time is it right now?")

run_tool_call("What is the capital of Japan?")