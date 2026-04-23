import os
import base64
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("MINIMAX_API_KEY", "").strip()
URL = "https://api.minimax.chat/v1/t2a_v2"

VOICE_MAP = {
    "Sae Chabashira": "moss_audio_9ffcdb94-3749-11f1-8aa8-1efaa00e25e8",
    "Haruka San": "moss_audio_01dcd58c-3664-11f1-8aa8-1efaa00e25e8",
    "Father Assistant": "moss_audio_a1cd6c15-3628-11f1-aeab-ca3bf5691d07",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Line(BaseModel):
    text: str
    voice: str

class Req(BaseModel):
    lines: List[Line]


# -------------------------
# CORE TTS CALL (SAFE)
# -------------------------
async def tts(text, voice_id, retry=True):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "speech-01",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(URL, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        logger.error(resp.text)
        return None

    # -------------------------
    # CASE 1: JSON RESPONSE
    # -------------------------
    try:
        data = resp.json()
        audio_b64 = data.get("data", {}).get("audio")

        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            if len(audio_bytes) < 1000:  # 🔥 prevents 0:00 silent audio
                return None
            return audio_bytes

    except Exception:
        pass

    # -------------------------
    # CASE 2: RAW AUDIO
    # -------------------------
    if len(resp.content) < 1000:
        return None

    return resp.content


# -------------------------
# API ENDPOINT
# -------------------------
@app.post("/generate-audio")
async def generate(req: Req):

    if not req.lines:
        raise HTTPException(400, "No input text")

    final_audio = []

    for line in req.lines:
        text = line.text.strip()
        if not text:
            continue

        voice_id = VOICE_MAP.get(line.voice)

        if not voice_id:
            raise HTTPException(400, f"Invalid voice: {line.voice}")

        # -------------------------
        # TRY 1
        # -------------------------
        audio = await tts(text, voice_id)

        # -------------------------
        # RETRY ON FAILURE
        # -------------------------
        if not audio:
            logger.warning("Retrying TTS...")
            audio = await tts(text, voice_id, retry=False)

        if not audio:
            raise HTTPException(
                500,
                f"Failed to generate audio for: '{text[:30]}...'"
            )

        final_audio.append(audio)

    if not final_audio:
        raise HTTPException(500, "No valid audio generated")

    # merge safely
    merged = b"".join(final_audio)

    if len(merged) < 1000:
        raise HTTPException(500, "Generated audio is too small (invalid)")

    b64 = base64.b64encode(merged).decode("utf-8")

    return {
        "audio_base64": f"data:audio/mp3;base64,{b64}"
    }


@app.get("/")
def health():
    return {"status": "OK"}
