import os
import json
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.stream_service import stream_capture_service

router = APIRouter(prefix="/api/capture", tags=["Stream Capture"])

SOURCES_FILE = "sources.json"
DEFAULT_SOURCES = [
    "https://cctv.jogjakota.go.id/malioboro/Malioboro_4_Kepatihan.stream/chunklist_w12345.m3u8"
]

class CaptureRequest(BaseModel):
    sources: List[str]
    num_frames: int = 200
    frame_skip: int = 2
    output_dir: str = "dataset_raw"

class SourcesSaveRequest(BaseModel):
    sources: List[str]

def load_sources_from_file() -> List[str]:
    if os.path.exists(SOURCES_FILE):
        try:
            with open(SOURCES_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            print(f"Error loading {SOURCES_FILE}: {e}")
    return DEFAULT_SOURCES

def save_sources_to_file(sources: List[str]):
    try:
        with open(SOURCES_FILE, "w") as f:
            json.dump(sources, f, indent=2)
    except Exception as e:
        print(f"Error saving {SOURCES_FILE}: {e}")

@router.get("/sources")
async def get_saved_sources():
    sources = load_sources_from_file()
    return {"sources": sources}

@router.post("/sources")
async def save_sources(req: SourcesSaveRequest):
    save_sources_to_file(req.sources)
    return {"status": "saved", "sources": req.sources}

@router.post("/start")
async def start_capture(req: CaptureRequest):
    if stream_capture_service.is_running:
        raise HTTPException(status_code=400, detail="Capture is already running.")

    if not req.sources:
        raise HTTPException(status_code=400, detail="At least one stream source must be provided.")

    # Save sources list to disk
    save_sources_to_file(req.sources)

    thread = threading.Thread(
        target=stream_capture_service.run_capture,
        args=(req.sources, req.num_frames, req.frame_skip, req.output_dir),
        daemon=True
    )
    thread.start()

    return {"status": "started", "message": f"Capture task initiated for {len(req.sources)} sources."}

@router.post("/cancel")
async def cancel_capture():
    stream_capture_service.cancel()
    return {"status": "cancelling", "message": "Cancel signal sent."}

@router.get("/status")
async def get_capture_status():
    return {
        "is_running": stream_capture_service.is_running,
        "progress": stream_capture_service.current_progress,
        "total": stream_capture_service.total_frames,
        "last_saved_frame": stream_capture_service.last_saved_frame,
        "logs": stream_capture_service.logs[-20:]
    }
