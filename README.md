# Agentic AI : 1-Week Internship Project

A 7-day build-up from raw LLM calls to a full RAG agent with tools, memory, and guardrails — running free and local via Ollama (Llama 3.1).

## What's inside

| Day | Focus |
|---|---|
| 1 | API basics, tokens, temperature, email classifier |
| 2 | Prompting techniques, structured extraction, LangChain |
| 3 | Embeddings, PDF chunking, ChromaDB, RAG Q&A bot |
| 4 | Tool calling, multi-step chaining, MCP |
| 5 | Agents (hand-written + LangGraph), memory, safety controls |
| 6 | Multi-agent system, prompt injection defense, eval suite, cost optimization |
| 7 | **Capstone:** RAG + tools + memory + guardrails, Streamlit UI |

## Stack

Python · Ollama (Llama 3.1) · LangChain / LangGraph · ChromaDB · sentence-transformers · Streamlit · Pydantic

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install langchain langchain-community langchain-ollama langchain-core langchain-text-splitters langgraph chromadb sentence-transformers pypdf streamlit numpy pydantic tiktoken
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Run the capstone app

```
streamlit run app.py
```

Chat with your PDFs (`doc1–3.pdf`), ask calculations, get cited sources — all in one UI.

## Guardrails included

- Iteration caps + timeouts (no runaway loops)
- Human approval before destructive actions
- Prompt injection defense
- Sensitive-info filter (blocks password/SSN/API key requests)

## Results

- ~90–100% accuracy on 25-case eval suite
- 80%+ cost reduction via concise prompting
- $0 spent (fully local)

