import os
import uuid
import logging
import shutil
from pathlib import Path
from typing import List
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_TTS_URL = "https://api.minimax.chat/v1/t2a_v2"

VOICE_MAP = {
    "Sae Chabashira": "moss_audio_9ffcdb94-3749-11f1-8aa8-1efaa00e25e8",
    "Haruka San": "moss_audio_01dcd58c-3664-11f1-bc6c-e264257f4e44",
    "Father Assistant": "moss_audio_a1cd6c15-3628-11f1-aeab-ca3bf5691d07",
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

class TextLine(BaseModel):
    text: str
    voice: str

class GenerateRequest(BaseModel):
    lines: List[TextLine]

class GenerateResponse(BaseModel):
    file_id: str
    download_url: str

async def call_minimax(text, voice_id):
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "speech-01", # Using version 01 for better compatibility
        "text": text,
        "stream": False,
        "voice_setting": {"voice_id": voice_id, "speed": 1.0, "vol": 1.0, "pitch": 0},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(MINIMAX_TTS_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            logger.error(f"Error: {resp.status_code} - {resp.text}")
            return None
        return resp.content

@app.post("/generate-audio", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not MINIMAX_API_KEY:
        raise HTTPException(500, "API Key Missing")

    session_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{session_id}.mp3"

    # TEST: Only process the FIRST line to avoid merging issues
    first_line = req.lines[0]
    voice_id = VOICE_MAP.get(first_line.voice, VOICE_MAP["Sae Chabashira"])
    
    audio_data = await call_minimax(first_line.text, voice_id)

    if audio_data is None or len(audio_data) == 0:
        logger.error("MiniMax returned NO data. Check your API Key!")
        raise HTTPException(500, "MiniMax API failed to return audio.")

    with open(output_path, "wb") as f:
        f.write(audio_data)

    return GenerateResponse(
        file_id=session_id,
        download_url=f"/outputs/{session_id}.mp3"
    )

@app.get("/")
def root():
    return {"status": "working"}
    
