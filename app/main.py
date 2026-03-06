"""
SingCoach MVP - FastAPI Application
Single service: serves frontend, reference audio, and analysis API.
"""

import os
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.analyzer import AudioAnalyzer
from app.coach import CoachingEngine

logger = logging.getLogger("singcoach")

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "chillipeppers_sample.mp3"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="SingCoach", version="0.1.0")

analyzer = AudioAnalyzer(sr=16000, hop_length=512, window_sec=0.5)
coach = CoachingEngine()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/reference")
async def get_reference():
    if not REFERENCE_PATH.exists():
        raise HTTPException(status_code=404, detail="Reference audio not found")
    return FileResponse(
        path=str(REFERENCE_PATH),
        media_type="audio/mpeg",
        filename="reference.mp3",
    )


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
            reference_path=str(REFERENCE_PATH),
            user_path=tmp_path,
        )
        feedback_items = coach.generate_feedback(segments)
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
