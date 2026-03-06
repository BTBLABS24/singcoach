"""
SingCoach MVP - FastAPI Application
Single service: serves frontend, reference audio, and analysis API.
Supports dynamic Whisper transcription and custom reference uploads.
"""

import os
import tempfile
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import whisper

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.analyzer import AudioAnalyzer
from app.coach import CoachingEngine

logger = logging.getLogger("singcoach")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = BASE_DIR / "chillipeppers_sample.mp3"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Mutable state: current reference track and its cached transcription
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

current_reference_path = str(DEFAULT_REFERENCE_PATH)
current_lyrics: List[Tuple[float, float, str]] = []

app = FastAPI(title="SingCoach", version="0.2.0")

analyzer = AudioAnalyzer(sr=16000, hop_length=512, window_sec=0.5)
coach = CoachingEngine()

# Load Whisper model (base is a good speed/accuracy trade-off for MVP)
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("base")
logger.info("Whisper model loaded.")


def transcribe_audio(audio_path: str) -> List[Tuple[float, float, str]]:
    """Transcribe audio file using Whisper, return list of (start, end, text)."""
    result = whisper_model.transcribe(audio_path, word_timestamps=False)
    lyrics = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:
            lyrics.append((seg["start"], seg["end"], text))
    return lyrics


@app.on_event("startup")
async def startup_transcribe():
    """Transcribe the default reference track on startup."""
    global current_lyrics
    if Path(current_reference_path).exists():
        try:
            current_lyrics = transcribe_audio(current_reference_path)
            logger.info(f"Transcribed default reference: {len(current_lyrics)} segments")
        except Exception as e:
            logger.warning(f"Failed to transcribe default reference: {e}")
            current_lyrics = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/reference")
async def get_reference():
    if not Path(current_reference_path).exists():
        raise HTTPException(status_code=404, detail="Reference audio not found")
    return FileResponse(
        path=current_reference_path,
        media_type="audio/mpeg",
        filename="reference.mp3",
    )


@app.get("/lyrics")
async def get_lyrics():
    """Return the current lyrics with timestamps."""
    return JSONResponse(content=[
        {"start": s, "end": e, "text": t}
        for s, e, t in current_lyrics
    ])


@app.post("/upload-reference")
async def upload_reference(reference_audio: UploadFile = File(...)):
    """Upload a custom acapella reference track. Replaces the current reference."""
    global current_reference_path, current_lyrics

    content = await reference_audio.read()
    if len(content) < 5000:
        raise HTTPException(status_code=400, detail="File too small — upload a valid audio file.")

    suffix = _get_suffix(reference_audio.filename)
    dest = UPLOADS_DIR / f"custom_reference{suffix}"

    with open(dest, "wb") as f:
        f.write(content)

    # Transcribe the new reference
    try:
        new_lyrics = transcribe_audio(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe uploaded audio: {e}")

    current_reference_path = str(dest)
    current_lyrics = new_lyrics

    return JSONResponse(content={
        "status": "ok",
        "lyrics": [{"start": s, "end": e, "text": t} for s, e, t in current_lyrics],
        "message": f"Reference updated. {len(current_lyrics)} lyric segments detected.",
    })


@app.post("/reset-reference")
async def reset_reference():
    """Reset to the default built-in reference track."""
    global current_reference_path, current_lyrics

    current_reference_path = str(DEFAULT_REFERENCE_PATH)

    if DEFAULT_REFERENCE_PATH.exists():
        try:
            current_lyrics = transcribe_audio(current_reference_path)
        except Exception:
            current_lyrics = []

    return JSONResponse(content={
        "status": "ok",
        "lyrics": [{"start": s, "end": e, "text": t} for s, e, t in current_lyrics],
    })


@app.post("/analyze")
async def analyze(user_audio: UploadFile = File(...)):
    content = await user_audio.read()
    if len(content) < 5000:
        raise HTTPException(status_code=400, detail="Recording too short — sing for at least a few seconds.")

    suffix = _get_suffix(user_audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments = analyzer.compare(
            reference_path=current_reference_path,
            user_path=tmp_path,
        )
        feedback_items = coach.generate_feedback(segments, lyrics=current_lyrics)
        return JSONResponse(content=feedback_items)

    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    finally:
        os.unlink(tmp_path)


def _get_suffix(filename: str | None) -> str:
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1]
    return ".webm"


# Mount static files LAST so API routes take priority
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
