import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# -----------------------------
# CORS MIDDLEWARE (IMPORTANT)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (safe for this assignment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CommentRequest(BaseModel):
    comment: str


sentiment_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "rating": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["sentiment", "rating"],
            "additionalProperties": False
        }
    }
}


@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis API. "
                        "Classify sentiment as positive, negative, or neutral. "
                        "Assign rating from 1 to 5: "
                        "1 = highly negative, "
                        "2 = negative, "
                        "3 = neutral, "
                        "4 = positive, "
                        "5 = highly positive. "
                        "Respond strictly according to the JSON schema."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format=sentiment_schema,
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))