import os
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from groq import Groq

# Load variables from .env
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


async def transcribe_audio_file(file: UploadFile):
    # 1. Check that a file was provided
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    # 2. Check that the uploaded file is an audio file
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be an audio file."
        )

    # 3. Read the audio file
    audio = await file.read()

    # 4. Check that the file isn't empty
    if len(audio) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty."
        )

    try:
        # Send audio to Groq Whisper
        transcription = client.audio.transcriptions.create(
            file=(file.filename, audio),
            model="whisper-large-v3-turbo"
        )

        return {
            "filename": file.filename,
            "text": transcription.text
        }

    except Exception as e:
        print("Groq transcription error:", e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while transcribing the audio."
        )
