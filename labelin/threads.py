import os
import cv2
import glob
import time
import shutil
from PyQt6.QtCore import QThread, pyqtSignal

class CaptureThread(QThread):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, sources, sb_num_frames, frame_skip, output_dir):
        super().__init__()
        self.sources = sources
        self.sb_num_frames = sb_num_frames
        self.frame_skip = frame_skip
        self.output_dir = output_dir

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        caps = []
        for src in self.sources:
            try:
                # Int cast for webcams if pure digit
                src_val = int(src) if src.isdigit() else src
                self.log.emit(f"Connecting to stream: {src_val}...")
                cap = cv2.VideoCapture(src_val)
                if cap.isOpened():
                    caps.append((src_val, cap))
                else:
                    self.log.emit(f"Failed to open stream: {src_val}")
            except Exception as e:
                self.log.emit(f"Error opening {src}: {e}")
                
        if not caps:
            self.finished.emit(False, "Failed to open any video streams.")
            return

        # Find highest frame number currently in the output directory
        start_idx = 0
        existing_frames = glob.glob(os.path.join(self.output_dir, "frame_*.jpg"))
        if existing_frames:
            indices = []
            for f in existing_frames:
                try:
                    basename = os.path.basename(f)
                    idx = int(basename.replace("frame_", "").replace(".jpg", ""))
                    indices.append(idx)
                except ValueError:
                    pass
            if indices:
                start_idx = max(indices) + 1
                
        counts = {id(cap): 0 for _, cap in caps}
        saved = 0
        self.log.emit(f"Starting capture from frame_{start_idx:04d}.jpg...")
        
        while saved < self.sb_num_frames.value() and caps:
            active_caps = []
            for src_name, cap in caps:
                if saved >= self.sb_num_frames.value():
                    active_caps.append((src_name, cap))
                    continue
                    
                ret, frame = cap.read()
                if ret:
                    active_caps.append((src_name, cap))
                    counts[id(cap)] += 1
                    if counts[id(cap)] % self.frame_skip == 0:
                        current_frame_id = start_idx + saved
                        filename = f"{self.output_dir}/frame_{current_frame_id:04d}.jpg"
                        cv2.imwrite(filename, frame)
                        saved += 1
                        self.progress.emit(saved, self.sb_num_frames.value())
                        self.log.emit(f"Saved frame {current_frame_id:04d} from {src_name} ({saved}/{self.sb_num_frames.value()})")
                else:
                    self.log.emit(f"Stream ended/disconnected: {src_name}")
                    cap.release()
            caps = active_caps
                
        for _, cap in caps:
            cap.release()
            
        self.finished.emit(True, f"Done! Saved {saved} frames to '{self.output_dir}'")


class AutoLabelThread(QThread):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, model_path, image_dir, output_dir):
        super().__init__()
        self.model_path = model_path
        self.image_dir = image_dir
        self.output_dir = output_dir

    def run(self):
        try:
            from ultralytics import YOLO
            self.log.emit(f"Loading model {self.model_path}...")
            model = YOLO(self.model_path)
            
            os.makedirs(f"{self.output_dir}/images", exist_ok=True)
            os.makedirs(f"{self.output_dir}/labels", exist_ok=True)
            
            images = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]
            total = len(images)
            
            if total == 0:
                self.finished.emit(False, "No images found to auto-label.")
                return
                
            for idx, img_name in enumerate(images):
                label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                manual_label_path = os.path.join("dataset_manual", "labels", label_name)
                
                if os.path.exists(manual_label_path) and os.path.getsize(manual_label_path) > 0:
                    self.log.emit(f"Skipping {img_name} (Already labeled)")
                    self.progress.emit(idx + 1, total)
                    continue
                    
                img_path = os.path.join(self.image_dir, img_name)
                frame = cv2.imread(img_path)
                
                results = model(frame, conf=0.2, iou=0.4, verbose=False)
                h, w = frame.shape[:2]
                labels = []
                
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls)
                        if cls in [2, 3, 5, 7]: # car, motor, bus, truck COCO
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            class_map = {2: 0, 3: 1, 5: 2, 7: 3}
                            new_cls = class_map[cls]
                            labels.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                if labels:
                    shutil.copy(img_path, f"{self.output_dir}/images/{img_name}")
                    label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                    with open(f"{self.output_dir}/labels/{label_name}", 'w') as f:
                        f.write('\n'.join(labels))
                
                self.progress.emit(idx + 1, total)
                
            self.finished.emit(True, f"Auto-labeled finished. Check '{self.output_dir}'.")
        except Exception as e:
            self.finished.emit(False, str(e))


class TrainThread(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, yaml_path, model_path, epochs, batch, imgsz, workers):
        super().__init__()
        self.yaml_path = yaml_path
        self.model_path = model_path
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.workers = workers

    def run(self):
        try:
            from ultralytics import YOLO
            self.log.emit(f"Loading YOLO model from {self.model_path}...")
            model = YOLO(self.model_path)
            
            self.log.emit(f"Starting training on {self.yaml_path}...")
            results = model.train(
                data=self.yaml_path,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                project='runs',
                name='yolo_traffic_gui',
                exist_ok=True,
                device=0,         # Force GPU 0
                workers=self.workers, # Reduces RAM to VRAM overhead
                amp=False         # Disable mixed precision to prevent NaN losses on GTX 16xx series
            )
            self.finished.emit(True, f"Training complete! Best model at runs/yolo_traffic_gui/weights/best.pt")
        except Exception as e:
            self.finished.emit(False, f"Training failed: {str(e)}")
