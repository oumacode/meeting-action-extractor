from fastapi import FastAPI, UploadFile, File
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    audio = await file.read()

    transcription = client.audio.transcriptions.create(
        file=(file.filename, audio),
        model="whisper-large-v3-turbo"
    )

    return {
        "text": transcription.text
    }