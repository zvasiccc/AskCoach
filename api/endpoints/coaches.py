import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.models import AskRequest
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from ingest.text_ingest import extract_text_from_pdf, ingest_raw_text
from db.chroma import ChromaDBManager
from agent.ask_question import ask_question

db = ChromaDBManager()
app = FastAPI()

@app.post("/ask")
async def ask(request: AskRequest):
    answer, context = ask_question(
        question=request.pitanje,
        coach_id=request.coach_id,
        history=request.history
    )
    return {
        "answer": answer,
        "context": context
    }

@app.post("/upload")
async def upload_knowledge(
    coach_id: str = Form(...),
    file: UploadFile = File(...)
):
    content = await file.read()
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    else:
        text = content.decode("utf-8")
    ingest_raw_text(text, coach_id, source_name=file.filename)
    return {"message": f"Baza '{coach_id}' je uspesno azurirana fajlom {file.filename}."}

#pomocne funkcije
@app.get("/coaches")
async def list_coaches():
    coaches = []
    for coach_id in db.list_coaches():
        collection = db.get_coach_collection(coach_id)
        coaches.append({"id": coach_id, "chunk_count": collection.count()})
    return {"coaches": coaches}

@app.get("/coaches/{coach_id}/chunks")
async def get_chunks(coach_id: str):
    collection = db.get_coach_collection(coach_id)
    results = collection.get()
    return {"chunks": results["documents"]}

@app.delete("/coaches/{coach_id}")
async def delete_coach(coach_id: str):
    success = db.delete_collection(coach_id)
    if success:
        return {"message": f"Baza za {coach_id} obrisana."}
    return {"message": "Greška pri brisanju."}, 400