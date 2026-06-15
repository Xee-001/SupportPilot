import chromadb
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("fastapi_docs")

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 75)

def ingest_docs():
    docs_path = Path("./fastapi/docs/en/docs")
    for doc in docs_path.glob("**/*.md"):
        with open(doc, "r") as f:
            text = f.read()
            chunks = splitter.split_text(text)
            embeddings = model.encode(chunks)
            collection.add(
                documents=chunks,
                metadatas=[{"source": str(doc)}] * len(chunks),
                ids=[f"{str(doc)}_{i}" for i in range(len(chunks))],
                embeddings=embeddings.tolist()
            )

if __name__ == "__main__":
    ingest_docs()            