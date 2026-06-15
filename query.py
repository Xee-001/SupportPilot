from dotenv import load_dotenv
import chromadb
from groq import Groq

from sentence_transformers import SentenceTransformer

load_dotenv()
llm = Groq()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("fastapi_docs")

def answer_question(question):
    embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=3)
    
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)

    sources = [m["source"] for m in results["metadatas"][0]]

    prompt = f"""You are a helpful assistant that answers questions based on the following context below.
    
    Context:
    {context}

    Question: {question}
    """

    response = llm.chat.completions.create(
        model ="llama-3.1-8b-instant",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt}
        ]

    )
    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    result = answer_question("What is FastAPI?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(s)
