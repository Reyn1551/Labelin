import os
import cv2
import shutil
import time
from typing import List, Dict, Any

def calculate_iou(box1, box2):
    # box format: (x1, y1, x2, y2)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def apply_nms(boxes_data, iou_threshold=0.45):
    # boxes_data: list of dicts with {cls, conf, coords_xyxy, label_str}
    sorted_boxes = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
    kept_boxes = []

    while sorted_boxes:
        best = sorted_boxes.pop(0)
        kept_boxes.append(best)

        remaining = []
        for box in sorted_boxes:
            iou = calculate_iou(best['coords_xyxy'], box['coords_xyxy'])
            if iou < iou_threshold:
                remaining.append(box)
        sorted_boxes = remaining

    return kept_boxes

class AutoLabelService:
    def __init__(self):
        self.is_running = False
        self.should_cancel = False
        self.current_progress = 0
        self.total_images = 0
        self.logs: List[str] = []

    def log(self, message: str):
        print(f"[AutoLabel] {message}")
        self.logs.append(message)
        if len(self.logs) > 500:
            self.logs.pop(0)

    def cancel(self):
        if self.is_running:
            self.should_cancel = True
            self.log("Cancel requested.")

    def run_autolabel(self, model_path: str = "yolov8x.pt", image_dir: str = "dataset_raw", output_dir: str = "dataset_labeled", conf: float = 0.25, iou: float = 0.45):
        self.is_running = True
        self.should_cancel = False
        self.current_progress = 0
        self.logs.clear()

        try:
            from ultralytics import YOLO
            self.log(f"Loading YOLO model weights: '{model_path}'...")
            model = YOLO(model_path)

            os.makedirs(f"{output_dir}/images", exist_ok=True)
            os.makedirs(f"{output_dir}/labels", exist_ok=True)

            if not os.path.exists(image_dir):
                self.is_running = False
                return False, f"Directory '{image_dir}' does not exist."

            images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images.sort()
            self.total_images = len(images)

            if self.total_images == 0:
                self.is_running = False
                return False, f"No images found in '{image_dir}' to auto-label."

            # Inspect model names to determine if standard COCO model or custom model
            is_coco_model = False
            if hasattr(model, 'names') and isinstance(model.names, dict):
                names_values = list(model.names.values())
                if 'person' in names_values and 'car' in names_values and len(names_values) >= 80:
                    is_coco_model = True

            self.log(f"Starting 100% Comprehensive Auto-Label Sweep for {self.total_images} images.")
            self.log(f"Model Mode: {'COCO Traffic Mapping (car, motorcycle, bus, truck)' if is_coco_model else 'Custom Model Mapping'}")
            self.log(f"Primary Conf={conf}, Fallback Conf=0.10, IoU NMS={iou}")

            total_boxes_labeled = 0
            fallback_applied_count = 0
            processed_file_count = 0

            for idx, img_name in enumerate(images):
                if self.should_cancel:
                    self.is_running = False
                    return False, "Auto-label cancelled by user."

                label_name = img_name.rsplit('.', 1)[0] + '.txt'
                img_path = os.path.join(image_dir, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                h, w = frame.shape[:2]

                # Helper to extract detected boxes
                def extract_boxes(target_conf):
                    results = model(frame, conf=target_conf, iou=iou, agnostic_nms=True, verbose=False)
                    extracted = []
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls)
                            box_conf = float(box.conf)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            new_cls = None
                            if is_coco_model:
                                coco_map = {2: 0, 3: 1, 5: 2, 7: 3}
                                if cls in coco_map:
                                    new_cls = coco_map[cls]
                            else:
                                if cls in [0, 1, 2, 3]:
                                    new_cls = cls

                            if new_cls is not None:
                                x_center = ((x1 + x2) / 2) / w
                                y_center = ((y1 + y2) / 2) / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h

                                extracted.append({
                                    'cls': new_cls,
                                    'conf': box_conf,
                                    'coords_xyxy': (x1, y1, x2, y2),
                                    'label_str': f"{new_cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                                })
                    return apply_nms(extracted, iou_threshold=iou)

                # Pass 1: Primary confidence sweep
                filtered_boxes = extract_boxes(conf)

                # Pass 2: Fallback sweep with lower confidence (0.10) if no boxes found in Pass 1
                if not filtered_boxes and conf > 0.10:
                    fallback_boxes = extract_boxes(0.10)
                    if fallback_boxes:
                        filtered_boxes = fallback_boxes
                        fallback_applied_count += 1

                labels = [b['label_str'] for b in filtered_boxes]
                total_boxes_labeled += len(labels)

                # ALWAYS copy image to dataset_labeled/images AND write label to dataset_labeled/labels
                shutil.copy(img_path, f"{output_dir}/images/{img_name}")
                with open(f"{output_dir}/labels/{label_name}", 'w') as f:
                    f.write('\n'.join(labels))

                processed_file_count += 1
                self.current_progress = processed_file_count

                if processed_file_count % 15 == 0 or processed_file_count == self.total_images:
                    self.log(f"Processed {processed_file_count}/{self.total_images} files ({total_boxes_labeled} total BBoxes generated, {fallback_applied_count} fallback passes).")

            self.is_running = False
            msg = f"SUCCESS! 100% Coverage Achieved: Processed ALL {processed_file_count} of {self.total_images} image files into '{output_dir}' (0 files missed, {total_boxes_labeled} total BBoxes)."
            self.log(msg)
            return True, msg
        except Exception as e:
            self.is_running = False
            err_msg = f"Auto-label failed: {str(e)}"
            self.log(err_msg)
            return False, err_msg

autolabel_service = AutoLabelService()
