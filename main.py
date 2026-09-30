from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from query import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        return answer_question(request.question)
    except Exception as e:
        print(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail="Could not answer the question. Please try again.")
