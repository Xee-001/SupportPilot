from fastapi import FastAPI
from pydantic import BaseModel

from query import answer_question

app =FastAPI()

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    result = answer_question(request.question)
    return result    