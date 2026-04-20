import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, UploadFile, File, Form
from ingest.text_ingest import ingest_raw_text 

app = FastAPI()

@app.post("/upload")
async def upload_knowledge(
    coach_id: str = Form(...), 
    file: UploadFile = File(...)
):
    # Pročitaj sadržaj fajla
    content = await file.read()
    text = content.decode("utf-8")
    
    print("text je",text)
    ingest_raw_text(text, coach_id, source_name=file.filename)
    
    return {"message": f"Baza za trenera '{coach_id}' je uspešno ažurirana fajlom {file.filename}."}