import os
import glob
import cv2
import shutil
import threading
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.autolabel_service import autolabel_service

router = APIRouter(prefix="/api/annotate", tags=["Annotation Canvas"])

class BBox(BaseModel):
    cls_id: int
    x_center: float
    y_center: float
    width: float
    height: float

class SaveAnnotationRequest(BaseModel):
    image_filename: str
    boxes: List[BBox]

class AutoLabelRequest(BaseModel):
    model_path: str = "yolov8x.pt"
    image_dir: str = "dataset_raw"
    output_dir: str = "dataset_labeled"
    conf: float = 0.25
    iou: float = 0.45

@router.get("/models")
async def list_models():
    os.makedirs("models", exist_ok=True)
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt", "yolo11n.pt", "yolo11s.pt", "yolo11x.pt"]

    # Check root directory for custom .pt files
    root_pts = [f for f in os.listdir(".") if f.endswith(".pt") and f not in models]
    models.extend(root_pts)

    # Check models/ directory
    custom_models = [os.path.join("models", f) for f in os.listdir("models") if f.endswith(".pt")]
    models.extend(custom_models)

    # Check runs/ detect training weights if any exist
    runs_pts = glob.glob("runs/**/*.pt", recursive=True)
    models.extend(runs_pts)

    return {"models": models}

@router.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only YOLO model files (.pt) are supported.")

    os.makedirs("models", exist_ok=True)
    target_path = os.path.join("models", file.filename)

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "model_path": target_path,
        "message": f"Successfully uploaded model '{file.filename}'."
    }

@router.get("/images")
async def list_images(folder: str = "unlabeled"):
    # Two English Directories: 'unlabeled' and 'labeled'
    image_set = set()

    if folder == "unlabeled":
        raw_dir = "dataset_raw"
        if os.path.exists(raw_dir):
            for f in os.listdir(raw_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_set.add(f)
    elif folder == "labeled":
        labeled_dir = "dataset_labeled/images"
        manual_dir = "dataset_manual/images"

        if os.path.exists(labeled_dir):
            for f in os.listdir(labeled_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_set.add(f)

        if os.path.exists(manual_dir):
            for f in os.listdir(manual_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_set.add(f)
    else:
        # Fallback for legacy folder names (dataset_raw, dataset_manual, dataset_labeled)
        if os.path.exists(folder):
            sub_img_dir = os.path.join(folder, "images") if os.path.exists(os.path.join(folder, "images")) else folder
            for f in os.listdir(sub_img_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_set.add(f)

    images = list(image_set)
    # Numerical sort for frame_0000.jpg, frame_0001.jpg, ..., frame_1235.jpg
    def sort_key(name):
        try:
            clean = name.replace("frame_", "").split('.')[0]
            return int(clean)
        except ValueError:
            return name

    images.sort(key=sort_key)
    return {"images": images, "total": len(images), "folder": folder}

@router.get("/image_file/{folder}/{filename}")
async def get_image_file(folder: str, filename: str):
    # Determine which file path to serve based on folder mode
    paths_to_try = []

    if folder == "labeled":
        paths_to_try = [
            os.path.join("dataset_manual", "images", filename),
            os.path.join("dataset_labeled", "images", filename),
            os.path.join("dataset_raw", filename)
        ]
    elif folder == "unlabeled":
        paths_to_try = [
            os.path.join("dataset_raw", filename),
            os.path.join("dataset_labeled", "images", filename),
            os.path.join("dataset_manual", "images", filename)
        ]
    else:
        paths_to_try = [
            os.path.join(folder, "images", filename),
            os.path.join(folder, filename),
            os.path.join("dataset_manual", "images", filename),
            os.path.join("dataset_labeled", "images", filename),
            os.path.join("dataset_raw", filename)
        ]

    for path in paths_to_try:
        if os.path.exists(path):
            return FileResponse(path)

    raise HTTPException(status_code=404, detail=f"Image file '{filename}' not found.")

@router.get("/labels/{folder}/{filename}")
async def get_label(folder: str, filename: str):
    label_filename = filename.rsplit('.', 1)[0] + '.txt'
    
    # Check dataset_manual first ONLY IF it exists and has size > 0
    manual_path = os.path.join("dataset_manual", "labels", label_filename)
    labeled_path = os.path.join("dataset_labeled", "labels", label_filename)

    label_path = None
    if os.path.exists(manual_path) and os.path.getsize(manual_path) > 0:
        label_path = manual_path
    elif os.path.exists(labeled_path) and os.path.getsize(labeled_path) > 0:
        label_path = labeled_path

    boxes = []
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        boxes.append({
                            "cls_id": cls_id,
                            "x_center": xc,
                            "y_center": yc,
                            "width": w,
                            "height": h
                        })
                    except ValueError:
                        pass

    return {"filename": filename, "label_filename": label_filename, "boxes": boxes}

@router.post("/save")
async def save_annotation(req: SaveAnnotationRequest):
    os.makedirs("dataset_manual/images", exist_ok=True)
    os.makedirs("dataset_manual/labels", exist_ok=True)

    label_filename = req.image_filename.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join("dataset_manual", "labels", label_filename)

    lines = []
    for box in req.boxes:
        lines.append(f"{box.cls_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}")

    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))

    raw_img_path = os.path.join("dataset_raw", req.image_filename)
    labeled_img_path = os.path.join("dataset_labeled", "images", req.image_filename)
    dest_img_path = os.path.join("dataset_manual", "images", req.image_filename)

    if os.path.exists(raw_img_path) and not os.path.exists(dest_img_path):
        cv2_img = cv2.imread(raw_img_path)
        if cv2_img is not None:
            cv2.imwrite(dest_img_path, cv2_img)
    elif os.path.exists(labeled_img_path) and not os.path.exists(dest_img_path):
        cv2_img = cv2.imread(labeled_img_path)
        if cv2_img is not None:
            cv2.imwrite(dest_img_path, cv2_img)

    return {"status": "saved", "message": f"Saved {len(req.boxes)} bounding boxes for {req.image_filename}."}

@router.post("/autolabel/start")
async def start_autolabel(req: AutoLabelRequest):
    if autolabel_service.is_running:
        raise HTTPException(status_code=400, detail="Auto-labeling is already in progress.")

    thread = threading.Thread(
        target=autolabel_service.run_autolabel,
        args=(req.model_path, req.image_dir, req.output_dir, req.conf, req.iou),
        daemon=True
    )
    thread.start()

    return {"status": "started", "message": "Auto-label task initiated."}

@router.post("/autolabel/cancel")
async def cancel_autolabel():
    autolabel_service.cancel()
    return {"status": "cancelling", "message": "Cancel signal sent."}

@router.get("/autolabel/status")
async def get_autolabel_status():
    return {
        "is_running": autolabel_service.is_running,
        "progress": autolabel_service.current_progress,
        "total": autolabel_service.total_images,
        "logs": autolabel_service.logs[-20:]
    }
