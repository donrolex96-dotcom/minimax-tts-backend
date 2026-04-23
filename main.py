import os
import uuid
import asyncio
import logging
import shutil
from pathlib import Path
from typing import List

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator # Back to v1 syntax
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")
MINIMAX_TTS_URL = "https://api.minimax.chat/v1/t2a_v2"

VOICE_MAP = {
    "Sae Chabashira": "moss_audio_9ffcdb94-3749-11f1-8aa8-1efaa00e25e8",
    "Haruka San": "moss_audio_01dcd58c-3664-11f1-bc6c-e264257f4e44",
    "Father Assistant": "moss_audio_a1cd6c15-3628-11f1-aeab-ca3bf5691d07",
}

OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MiniMax Multi-Voice TTS", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


class TextLine(BaseModel):
    text: str
    voice: str

    @validator("text") # Fixed for Pydantic v1 compatibility
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v


class GenerateRequest(BaseModel):
    lines: List[TextLine]


class GenerateResponse(BaseModel):
    file_id: str
    download_url: str


async def call_tts(client, text, voice_id):
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "speech-02-hd",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
    }

    url = MINIMAX_TTS_URL
    if MINIMAX_GROUP_ID:
        url += f"?GroupId={MINIMAX_GROUP_ID}"

    resp = await client.post(url, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        logger.error(f"TTS API Error: {resp.text}")
        raise RuntimeError(f"TTS failed: {resp.text[:200]}")

    return resp.content


async def merge_simple(files, output_path):
    if not files:
        return
    if len(files) == 1:
        shutil.copy(files[0], output_path)
        return

    with open(output_path, "wb") as w:
        for f in files:
            with open(f, "rb") as r:
                w.write(r.read())


def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


@app.post("/generate-audio", response_model=GenerateResponse)
async def generate(req: GenerateRequest, bg: BackgroundTasks):
    if not MINIMAX_API_KEY:
        raise HTTPException(500, "Missing API key")

    session = uuid.uuid4().hex
    session_dir = TEMP_DIR / session
    session_dir.mkdir(parents=True, exist_ok=True)

    files = []

    try:
        async with httpx.AsyncClient() as client:
            for i, line in enumerate(req.lines):
                voice_id = VOICE_MAP.get(line.voice)
                if not voice_id:
                    raise HTTPException(400, f"Invalid voice: {line.voice}")

                audio = await call_tts(client, line.text, voice_id)

                path = session_dir / f"{i}.mp3"
                path.write_bytes(audio)
                files.append(path)

        output = OUTPUT_DIR / f"{session}.mp3"
        await merge_simple(files, output)
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        cleanup(session_dir)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(500, f"Processing failed: {str(e)}")

    bg.add_task(cleanup, session_dir)

    return GenerateResponse(
        file_id=session,
        download_url=f"/outputs/{session}.mp3"
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-Voice TTS is Ready"}


@app.get("/health")
def health():
    return {"ok": True}
    
