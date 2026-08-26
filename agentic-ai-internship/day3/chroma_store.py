
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=30)

all_chunks = []
all_metadata = []

for pdf in pdf_files:
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    chunks = splitter.split_text(text)
    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadata.append({"source": pdf})  # metadata = which file it came from

print(f"Total chunks across all PDFs: {len(all_chunks)}")

client = chromadb.Client()
collection = client.create_collection(name="my_documents")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadata,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

print("All chunks stored in Chroma!\n")

query = "How has private industry changed the cost of space travel?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"source": "doc1.pdf"}  
)

print(f"Query: {query}")
print("Results (filtered to doc1.pdf only):\n")
for doc, meta, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    print(f"Source: {meta['source']} | Distance: {distance:.4f}")
    print(doc[:200], "...\n")
