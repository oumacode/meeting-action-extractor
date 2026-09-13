from fastapi import FastAPI, UploadFile, File, Body
from services.text_extraction_service import transcribe_audio_file
from services.task_extraction_service import extract_tasks_from_text

# Create FastAPI application
app = FastAPI(title="Meeting Action Extractor")


@app.get("/")
def root():
    return {"message": "Hello World"}

# Speech-to-text endpoint
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    return await transcribe_audio_file(file)


# Task extraction endpoint
@app.post("/extract-tasks")
async def extract_tasks(text: str = Body(..., embed=True)):
    return await extract_tasks_from_text(text)
