from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

all_docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    pages = loader.load()
    all_docs.extend(pages)

print(f"Loaded {len(all_docs)} pages total from {len(pdf_files)} PDFs")

splitter_small = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=30
)
splitter_large = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800, chunk_overlap=80
)

chunks_small = splitter_small.split_documents(all_docs)
chunks_large = splitter_large.split_documents(all_docs)

print(f"Small chunks (300 tokens): {len(chunks_small)} chunks")
print(f"Large chunks (800 tokens): {len(chunks_large)} chunks")

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts)
    return texts, embeddings

texts_small, emb_small = embed_chunks(chunks_small)
texts_large, emb_large = embed_chunks(chunks_large)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top3(query, texts, embeddings):
    query_emb = model.encode(query)
    scores = [(cosine_similarity(query_emb, e), t) for e, t in zip(embeddings, texts)]
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:3]

query = "How has private industry changed the cost of space travel?" 

print(f"\n=== QUERY: {query} ===")

print("\n--- Top 3 results from SMALL chunks (300 tokens) ---")
for score, text in retrieve_top3(query, texts_small, emb_small):
    print(f"\nScore: {score:.4f}")
    print(text[:200], "...")

print("\n--- Top 3 results from LARGE chunks (800 tokens) ---")
for score, text in retrieve_top3(query, texts_large, emb_large):
    print(f"\nScore: {score:.4f}")
    print(text[:200], "...")
