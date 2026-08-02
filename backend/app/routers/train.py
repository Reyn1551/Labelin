import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
from app.services.train_service import train_service

router = APIRouter(prefix="/api/train", tags=["YOLO Training"])

class TrainRequest(BaseModel):
    yaml_path: str = "dataset/traffic.yaml"
    model_path: str = "yolo11n.pt"
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    workers: int = 2

@router.post("/start")
async def start_training(req: TrainRequest):
    success, msg = train_service.start_training(
        yaml_path=req.yaml_path,
        model_path=req.model_path,
        epochs=req.epochs,
        batch=req.batch,
        imgsz=req.imgsz,
        workers=req.workers
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "started", "message": msg}

@router.get("/status")
async def get_training_status():
    return {
        "is_running": train_service.is_running,
        "metrics": train_service.latest_metrics,
        "recent_logs": train_service.logs[-30:]
    }

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue()
    train_service.add_listener(queue)

    # Send initial existing logs
    for log_item in train_service.logs[-50:]:
        await websocket.send_text(log_item)

    try:
        while True:
            log_line = await queue.get()
            await websocket.send_text(log_line)
    except WebSocketDisconnect:
        train_service.remove_listener(queue)
    except Exception:
        train_service.remove_listener(queue)
