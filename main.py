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
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")
MINIMAX_TTS_URL = "https://api.minimax.chat/v1/t2a_v2"

VOICE_MAP = {
    "Sae Chabashira":  "moss_audio_9ffcdb94-3749-11f1-8aa8-1efaa00e25e8",
    "Haruka San":      "moss_audio_01dcd58c-3664-11f1-bc6c-e264257f4e44",
    "Father Assistant":"moss_audio_a1cd6c15-3628-11f1-aeab-ca3bf5691d07",
}

SILENCE_MS = 400
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MiniMax Multi-Voice TTS", version="1.0.0")

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

class TextLine(BaseModel):
    text: str
    voice: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v

    @field_validator("voice")
    @classmethod
    def voice_must_be_valid(cls, v):
        if v not in VOICE_MAP:
            raise ValueError(f"voice '{v}' is not valid")
        return v

class GenerateRequest(BaseModel):
    lines: List[TextLine]

    @field_validator("lines")
    @classmethod
    def lines_not_empty(cls, v):
        if not v:
            raise ValueError("lines must contain at least one entry")
        if len(v) > 50:
            raise ValueError("maximum 50 lines per request")
        return v

class GenerateResponse(BaseModel):
    file_id: str
    download_url: str
    duration_hint: str

async def call_minimax_tts(client: httpx.AsyncClient, text: str, voice_id: str, attempt: int = 1) -> bytes:
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY is not set")

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
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
    }

    url = MINIMAX_TTS_URL
    if MINIMAX_GROUP_ID:
        url = f"{MINIMAX_TTS_URL}?GroupId={MINIMAX_GROUP_ID}"

    logger.info(f"[TTS] Attempt {attempt}: voice={voice_id}, text_len={len(text)}")
    resp = await client.post(url, headers=headers, json=payload, timeout=60.0)

    if resp.status_code != 200:
        body = resp.text[:500]
        logger.error(f"[TTS] HTTP {resp.status_code}: {body}")
        raise RuntimeError(f"TTS API returned {resp.status_code}: {body}")

    content_type = resp.headers.get("content-type", "")

    if "application/json" in content_type:
        data = resp.json()
        audio_hex = data.get("data", {}).get("audio", "")
        if audio_hex:
            return bytes.fromhex(audio_hex)
        import base64
        audio_b64 = data.get("data", {}).get("audio", "")
        if audio_b64:
            return base64.b64decode(audio_b64)
        raise ValueError(f"Unexpected JSON response: {str(data)[:300]}")

    return resp.content

async def fetch_audio_with_retry(client: httpx.AsyncClient, text: str, voice_id: str) -> bytes:
    try:
        return await call_minimax_tts(client, text, voice_id, attempt=1)
    except Exception as e:
        logger.warning(f"[TTS] First attempt failed ({e}), retrying...")
        await asyncio.sleep(1.5)
        return await call_minimax_tts(client, text, voice_id, attempt=2)

async def merge_audio_files(clip_paths: List[Path], output_path: Path, silence_ms: int = SILENCE_MS):
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], output_path)
        return

    silence_path = output_path.parent / f"silence_{uuid.uuid4().hex}.mp3"
    silence_sec = silence_ms / 1000.0

    silence_cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate=32000",
        "-t", str(silence_sec),
        "-codec:a", "libmp3lame", "-b:a", "128k",
        str(silence_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *silence_cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Failed to generate silence clip")

    interleaved = []
    for i, clip in enumerate(clip_paths):
        interleaved.append(clip)
        if i < len(clip_paths) - 1:
            interleaved.append(silence_path)

    concat_file = output_path.parent / f"concat_{uuid.uuid4().hex}.txt"
    with open(concat_file, "w") as f:
        for p in interleaved:
            f.write(f"file '{p.resolve()}'\n")

    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-codec:a", "libmp3lame", "-b:a", "128k", "-ar", "32000",
        str(output_path),
    ]

    proc2 = await asyncio.create_subprocess_exec(
        *concat_cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr2 = await proc2.communicate()

    silence_path.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)

    if proc2.returncode != 0:
        raise RuntimeError("Failed to merge audio clips")

def cleanup_temp_dir(session_dir: Path):
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[cleanup] Failed: {e}")

@app.post("/generate-audio", response_model=GenerateResponse)
async def generate_audio(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not MINIMAX_API_KEY:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY not set")

    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []

    async with httpx.AsyncClient() as client:
        for idx, line in enumerate(req.lines):
            voice_id = VOICE_MAP[line.voice]
            try:
                audio_bytes = await fetch_audio_with_retry(client, line.text, voice_id)
            except Exception as e:
                cleanup_temp_dir(session_dir)
                raise HTTPException(status_code=502, detail=f"TTS failed for line {idx+1}: {str(e)}")

            clip_path = session_dir / f"clip_{idx:03d}.mp3"
            clip_path.write_bytes(audio_bytes)
            clip_paths.append(clip_path)

    output_filename = f"output_{session_id}.mp3"
    output_path = OUTPUT_DIR / output_filename

    try:
        await merge_audio_files(clip_paths, output_path)
    except Exception as e:
        cleanup_temp_dir(session_dir)
        raise HTTPException(status_code=500, detail=f"Audio merge failed: {str(e)}")

    background_tasks.add_task(cleanup_temp_dir, session_dir)

    return GenerateResponse(
        file_id=session_id,
        download_url=f"/outputs/{output_filename}",
        duration_hint=f"~{len(req.lines) * 3}s estimated",
    )

@app.get("/voices")
def list_voices():
    return {"voices": list(VOICE_MAP.keys())}

@app.get("/health")
def health():
    return {"status": "ok", "api_key_set": bool(MINIMAX_API_KEY)}

@app.get("/")
def root():
    return {"service": "MiniMax Multi-Voice TTS API", "version": "1.0.0"}
