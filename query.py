import re
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import chromadb
from groq import Groq

from sentence_transformers import SentenceTransformer

load_dotenv()
llm = Groq()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("fastapi_docs")

all_docs= collection.get()  #fetches all documents in the collection(chroma)
corpus = all_docs["documents"] # list of all the text chunks in the collection
ids = all_docs["ids"]

def tokenize(text):
    return re.findall(r"\w+", text.lower())

tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


def answer_question(question):
    embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=10)
    
    tokenized_query = tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_ids = [ids[i] for i in bm25_top_indices]

    vector_ids = results["ids"][0]
    rrf_scores = {}
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/(rank + 60)
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/(rank + 60)
    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:3]

    id_to_chunk = dict(zip(ids, corpus))
    final_chunks = [id_to_chunk[i] for i in top_ids]
    context = "\n\n".join(final_chunks)


    sources = list(dict.fromkeys([id.rsplit("_", 1)[0] for id in top_ids]))


    prompt = f"""You are a helpful assistant that answers questions based on the following context below. If the answer is not in the context, say you don't know.
    
    Context:
    {context}

    Question: {question}
    """

    response = llm.chat.completions.create(
        model ="openai/gpt-oss-20b",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]

    )
    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}



if __name__ == "__main__":
    result = answer_question("How do I declare a query parameter?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(s)
