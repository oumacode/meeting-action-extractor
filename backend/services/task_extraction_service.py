import json
import os

from dotenv import load_dotenv
from fastapi import HTTPException
from groq import Groq


# Load variables from .env
load_dotenv()


# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


async def extract_tasks_from_text(text: str):

    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text provided for task extraction."
        )

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI assistant specialized in extracting action items
from meeting transcripts.

Extract the following information:

- action: the task that needs to be done
- responsible: the person responsible for the task, if mentioned
- deadline: the deadline, if mentioned

Important rules:

1. Do not invent information.
2. If the responsible person is not mentioned, return null.
3. If the deadline is not mentioned, return null.
4. Keep the action concise.
"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_action",
                    "strict": True,
                    "schema": {
                        "type": "object",

                        "properties": {
                            "action": {
                                "type": "string"
                            },
                            "responsible": {
                                "type": ["string", "null"]
                            },
                            "deadline": {
                                "type": ["string", "null"]
                            }
                        },

                        "required": [
                            "action",
                            "responsible",
                            "deadline"
                        ],

                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(
            response.choices[0].message.content
        )

        return {
            "text": text,
            "task": result
        }

    except Exception as e:

        print("Groq task extraction error:", e)

        raise HTTPException(
            status_code=500,
            detail="An error occurred while extracting tasks from text."
        )