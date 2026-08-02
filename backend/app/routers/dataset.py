from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.dataset_service import dataset_service

router = APIRouter(prefix="/api/dataset", tags=["Dataset Preparation"])

class SplitRequest(BaseModel):
    source_dir: str = "dataset_manual"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    base_dest: str = "dataset"
    class_names: Optional[List[str]] = ["car", "motorcycle", "bus", "truck"]

@router.post("/split")
async def split_dataset_endpoint(req: SplitRequest):
    success, msg, summary = dataset_service.split_dataset(
        source_dir=req.source_dir,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        test_ratio=req.test_ratio,
        base_dest=req.base_dest,
        class_names=req.class_names or ["car", "motorcycle", "bus", "truck"]
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg, "summary": summary}
