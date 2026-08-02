import sys
import os
import glob
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, QSize, QSizeF, pyqtSignal, QThread
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap, QImage, QFont,
    QKeySequence, QShortcut, QIcon, QTransform, QCursor
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QFileDialog, QFrame, QSpinBox, QDoubleSpinBox,
    QMessageBox, QStatusBar, QProgressBar, QDialog, QSlider, QTextEdit
)

# Standard Traffic Classes & Color Tokens
CLASSES = [
    {"id": 0, "name": "car", "color": QColor("#3B82F6")},
    {"id": 1, "name": "motorcycle", "color": QColor("#10B981")},
    {"id": 2, "name": "bus", "color": QColor("#F59E0B")},
    {"id": 3, "name": "truck", "color": QColor("#EC4899")}
]

# IoU calculation for NMS deduplication
def compute_iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0

def apply_class_agnostic_nms(boxes, iou_threshold=0.40):
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    keep = []
    while sorted_boxes:
        current = sorted_boxes.pop(0)
        keep.append(current)
        sorted_boxes = [
            b for b in sorted_boxes
            if compute_iou_xyxy(current['coords_xyxy'], b['coords_xyxy']) < iou_threshold
        ]
    return keep


class AutoLabelWorker(QThread):
    progress_changed = pyqtSignal(int, int, str) # current, total, log_message
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_path, image_dir, output_dir, conf=0.20, iou=0.40):
        super().__init__()
        self.model_path = model_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.conf = conf
        self.iou = iou
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            from ultralytics import YOLO
            self.progress_changed.emit(0, 0, f"Loading YOLO model weights '{self.model_path}'...")
            model = YOLO(self.model_path)

            os.makedirs(f"{self.output_dir}/images", exist_ok=True)
            os.makedirs(f"{self.output_dir}/labels", exist_ok=True)

            if not os.path.exists(self.image_dir):
                self.finished_signal.emit(False, f"Directory '{self.image_dir}' does not exist.")
                return

            images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            def sort_key(name):
                try:
                    clean = name.replace("frame_", "").split('.')[0]
                    return int(clean)
                except ValueError:
                    return name
            images.sort(key=sort_key)
            total_images = len(images)

            if total_images == 0:
                self.finished_signal.emit(False, f"No images found in '{self.image_dir}'.")
                return

            is_coco_model = False
            if hasattr(model, 'names') and isinstance(model.names, dict):
                names_values = list(model.names.values())
                if 'person' in names_values and 'car' in names_values and len(names_values) >= 80:
                    is_coco_model = True

            self.progress_changed.emit(0, total_images, f"Started Auto-Labeling for {total_images} images (Conf={self.conf:.2f}, IoU={self.iou:.2f})...")

            total_boxes_labeled = 0

            for idx, img_name in enumerate(images):
                if self.is_cancelled:
                    self.finished_signal.emit(False, "Auto-labeling cancelled by user.")
                    return

                label_name = img_name.rsplit('.', 1)[0] + '.txt'
                img_path = os.path.join(self.image_dir, img_name)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                h, w = frame.shape[:2]

                def extract_boxes(target_conf):
                    results = model(frame, conf=target_conf, iou=self.iou, agnostic_nms=True, verbose=False)
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
                                x_center = ((x1 + x2) / 2.0) / w
                                y_center = ((y1 + y2) / 2.0) / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h

                                extracted.append({
                                    'cls': new_cls,
                                    'conf': box_conf,
                                    'coords_xyxy': (x1, y1, x2, y2),
                                    'label_str': f"{new_cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                                })
                    return apply_class_agnostic_nms(extracted, iou_threshold=self.iou)

                # Pass 1: Primary confidence sweep
                filtered_boxes = extract_boxes(self.conf)

                # Pass 2: Fallback sweep (0.10) if no boxes found in Pass 1
                if not filtered_boxes and self.conf > 0.10:
                    fallback_boxes = extract_boxes(0.10)
                    if fallback_boxes:
                        filtered_boxes = fallback_boxes

                labels = [b['label_str'] for b in filtered_boxes]
                total_boxes_labeled += len(labels)

                # Write label text file
                out_label_path = os.path.join(self.output_dir, "labels", label_name)
                with open(out_label_path, 'w') as f:
                    f.write('\n'.join(labels))

                # Save image to dataset_labeled/images
                out_img_path = os.path.join(self.output_dir, "images", img_name)
                if not os.path.exists(out_img_path):
                    cv2.imwrite(out_img_path, frame)

                if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
                    msg = f"Auto-labeled [{idx+1}/{total_images}] {img_name} -> {len(labels)} boxes"
                    self.progress_changed.emit(idx + 1, total_images, msg)

            self.finished_signal.emit(True, f"SUCCESS! Processed ALL {total_images} images ({total_boxes_labeled} total bounding boxes created).")

        except Exception as e:
            self.finished_signal.emit(False, f"Auto-labeling Error: {str(e)}")


class AutoLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚡ YOLO AI Auto-Label Assistant")
        self.resize(560, 480)
        self.worker = None

        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Title
        title_lbl = QLabel("⚡ YOLO Traffic Auto-Labeling Engine")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #A855F7;")
        layout.addWidget(title_lbl)

        desc_lbl = QLabel("Automatically generate high-accuracy bounding boxes for all raw frames in 'dataset_raw' using heavy YOLO models (e.g. YOLOv8x).")
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        layout.addWidget(desc_lbl)

        # Model Selector
        model_layout = QHBoxLayout()
        model_lbl = QLabel("Model Weights:")
        model_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        model_layout.addWidget(model_lbl)

        self.model_combo = QComboBox()
        models = ["yolov8x.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11x.pt"]
        for f in os.listdir("."):
            if f.endswith(".pt") and f not in models:
                models.append(f)
        self.model_combo.addItems(models)
        model_layout.addWidget(self.model_combo, 1)

        self.browse_model_btn = QPushButton("Browse .pt...")
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.browse_model_btn)

        layout.addLayout(model_layout)

        # Preset Quick Button
        preset_layout = QHBoxLayout()
        preset_lbl = QLabel("Threshold Settings:")
        preset_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        preset_layout.addWidget(preset_lbl)

        self.preset_btn = QPushButton("⚡ Optimal Preset (Conf 0.20, IoU 0.40)")
        self.preset_btn.setStyleSheet("background: #0284C7; color: #FFF; font-weight: bold;")
        self.preset_btn.clicked.connect(self.apply_optimal_preset)
        preset_layout.addWidget(self.preset_btn)
        layout.addLayout(preset_layout)

        # Sliders Frame
        sliders_frame = QFrame()
        sliders_frame.setStyleSheet("background: #0B0F19; border: 1px solid #1F2937; border-radius: 8px; padding: 10px;")
        sliders_layout = QVBoxLayout(sliders_frame)

        # Conf Slider
        conf_header = QHBoxLayout()
        conf_header.addWidget(QLabel("Confidence Threshold (Conf):"))
        self.conf_val_lbl = QLabel("0.20")
        self.conf_val_lbl.setStyleSheet("color: #38BDF8; font-weight: bold; font-family: monospace;")
        conf_header.addWidget(self.conf_val_lbl)
        sliders_layout.addLayout(conf_header)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 80)
        self.conf_slider.setValue(20)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_val_lbl.setText(f"{v/100:.2f}"))
        sliders_layout.addWidget(self.conf_slider)

        # IoU Slider
        iou_header = QHBoxLayout()
        iou_header.addWidget(QLabel("IoU Overlap NMS Threshold:"))
        self.iou_val_lbl = QLabel("0.40")
        self.iou_val_lbl.setStyleSheet("color: #A855F7; font-weight: bold; font-family: monospace;")
        iou_header.addWidget(self.iou_val_lbl)
        sliders_layout.addLayout(iou_header)

        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(10, 80)
        self.iou_slider.setValue(40)
        self.iou_slider.valueChanged.connect(lambda v: self.iou_val_lbl.setText(f"{v/100:.2f}"))
        sliders_layout.addWidget(self.iou_slider)

        layout.addWidget(sliders_frame)

        # Progress Bar & Logs
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #06080E; color: #94A3B8; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_text, 1)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Start Auto-Labeling")
        self.start_btn.setStyleSheet("background: linear-gradient(135deg, #8B5CF6, #6D28D9); font-weight: bold; color: #FFF; padding: 10px; font-size: 13px;")
        self.start_btn.clicked.connect(self.start_autolabel)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel / Close")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #111827;
                color: #F8FAFC;
                font-family: 'Inter', sans-serif;
            }
            QComboBox, QPushButton {
                background-color: #1F2937;
                color: #F8FAFC;
                border: 1px solid #374151;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #374151;
            }
            QProgressBar {
                border: 1px solid #374151;
                border-radius: 5px;
                text-align: center;
                background-color: #0B0F19;
                color: #FFF;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #A855F7;
            }
        """)

    def browse_model_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model Weights", "", "YOLO Models (*.pt)")
        if fn:
            self.model_combo.addItem(fn)
            self.model_combo.setCurrentText(fn)

    def apply_optimal_preset(self):
        self.conf_slider.setValue(20)
        self.iou_slider.setValue(40)

    def start_autolabel(self):
        model_path = self.model_combo.currentText()
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0

        self.start_btn.setEnabled(False)
        self.log_text.append(f"[START] Model: {model_path} | Conf: {conf:.2f} | IoU: {iou:.2f}")

        self.worker = AutoLabelWorker(model_path, "dataset_raw", "dataset_labeled", conf, iou)
        self.worker.progress_changed.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, current, total, msg):
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
        self.log_text.append(msg)

    def on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.log_text.append(f"[{'FINISHED' if success else 'ERROR'}] {message}")
        if success:
            QMessageBox.information(self, "Auto-Label Complete", message)
            if self.parent() and hasattr(self.parent(), 'load_directory'):
                self.parent().load_directory("dataset_labeled")
            self.accept()
        else:
            QMessageBox.warning(self, "Auto-Label Failed", message)


class NativeAnnotationCanvas(QWidget):
    box_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.img_pixmap = None
        self.img_size = QSize(0, 0)

        self.boxes = []
        self.selected_box_idx = -1
        self.selected_class_id = 0

        # Tool Modes: 'draw', 'select'
        self.tool_mode = 'draw'

        # Zoom & Pan State
        self.zoom_level = 1.0
        self.max_zoom = 5.0
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.pan_start = QPointF(0, 0)

        # Drawing State
        self.is_drawing = False
        self.draw_start_img = QPointF(0, 0)
        self.current_draw_rect_img = QRectF()
        self.mouse_img_pos = QPointF(-1, -1)

        # History for Undo/Redo
        self.history = []
        self.history_idx = -1

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            self.img_pixmap = None
            self.img_size = QSize(0, 0)
            self.update()
            return False

        pix = QPixmap(img_path)
        if pix.isNull():
            self.img_pixmap = None
            self.img_size = QSize(0, 0)
            self.update()
            return False

        self.img_pixmap = pix
        self.img_size = pix.size()
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.selected_box_idx = -1
        self.update()
        return True

    def set_boxes(self, boxes):
        self.boxes = list(boxes)
        self.selected_box_idx = -1
        self.history = [list(boxes)]
        self.history_idx = 0
        self.update()

    def push_history(self, new_boxes):
        self.boxes = list(new_boxes)
        self.history = self.history[:self.history_idx + 1]
        self.history.append(list(new_boxes))
        self.history_idx += 1
        self.box_changed.emit()
        self.update()

    def undo(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.boxes = list(self.history[self.history_idx])
            self.selected_box_idx = -1
            self.box_changed.emit()
            self.update()

    def redo(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.boxes = list(self.history[self.history_idx])
            self.selected_box_idx = -1
            self.box_changed.emit()
            self.update()

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.update()

    def clamp_pan_offset(self):
        if not self.img_pixmap or self.img_size.isEmpty() or self.zoom_level <= 1.0:
            self.pan_offset = QPointF(0, 0)
            return

        cw = self.width() / 2.0
        ch = self.height() / 2.0
        iw = self.img_size.width()
        ih = self.img_size.height()

        fit_scale = min(self.width() / iw, self.height() / ih)
        scaled_w = iw * fit_scale * self.zoom_level
        scaled_h = ih * fit_scale * self.zoom_level

        max_extra_x = max(0.0, (scaled_w - self.width()) / 2.0)
        max_extra_y = max(0.0, (scaled_h - self.height()) / 2.0)

        px = max(-max_extra_x, min(max_extra_x, self.pan_offset.x()))
        py = max(-max_extra_y, min(max_extra_y, self.pan_offset.y()))
        self.pan_offset = QPointF(px, py)

    def widget_to_image_coords(self, pt: QPointF) -> QPointF:
        if not self.img_pixmap or self.img_size.isEmpty():
            return QPointF(-1, -1)

        cw = self.width() / 2.0
        ch = self.height() / 2.0
        iw = self.img_size.width()
        ih = self.img_size.height()

        fit_scale = min(self.width() / iw, self.height() / ih)
        scaled_w = iw * fit_scale * self.zoom_level
        scaled_h = ih * fit_scale * self.zoom_level

        img_top_left_x = cw - (scaled_w / 2.0) + self.pan_offset.x()
        img_top_left_y = ch - (scaled_h / 2.0) + self.pan_offset.y()

        img_x = (pt.x() - img_top_left_x) / (fit_scale * self.zoom_level)
        img_y = (pt.y() - img_top_left_y) / (fit_scale * self.zoom_level)

        return QPointF(img_x, img_y)

    def image_to_widget_pt(self, img_pt: QPointF) -> QPointF:
        if not self.img_pixmap or self.img_size.isEmpty():
            return QPointF(0, 0)

        cw = self.width() / 2.0
        ch = self.height() / 2.0
        iw = self.img_size.width()
        ih = self.img_size.height()

        fit_scale = min(self.width() / iw, self.height() / ih)
        effective_scale = fit_scale * self.zoom_level

        img_top_left_x = cw - (iw * effective_scale / 2.0) + self.pan_offset.x()
        img_top_left_y = ch - (ih * effective_scale / 2.0) + self.pan_offset.y()

        wx = img_top_left_x + (img_pt.x() * effective_scale)
        wy = img_top_left_y + (img_pt.y() * effective_scale)

        return QPointF(wx, wy)

    def image_to_widget_rect(self, rect_img: QRectF) -> QRectF:
        if not self.img_pixmap or self.img_size.isEmpty():
            return QRectF()

        cw = self.width() / 2.0
        ch = self.height() / 2.0
        iw = self.img_size.width()
        ih = self.img_size.height()

        fit_scale = min(self.width() / iw, self.height() / ih)
        effective_scale = fit_scale * self.zoom_level

        img_top_left_x = cw - (iw * effective_scale / 2.0) + self.pan_offset.x()
        img_top_left_y = ch - (ih * effective_scale / 2.0) + self.pan_offset.y()

        wx = img_top_left_x + (rect_img.x() * effective_scale)
        wy = img_top_left_y + (rect_img.y() * effective_scale)
        ww = rect_img.width() * effective_scale
        wh = rect_img.height() * effective_scale

        return QRectF(wx, wy, ww, wh)

    def wheelEvent(self, event):
        if not self.img_pixmap or self.img_size.isEmpty():
            event.accept()
            return

        cursor_widget_pt = event.position()
        img_pt_before = self.widget_to_image_coords(cursor_widget_pt)

        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom = min(self.max_zoom, self.zoom_level * 1.15)
        else:
            new_zoom = max(1.0, self.zoom_level / 1.15)

        if new_zoom == self.zoom_level:
            event.accept()
            return

        self.zoom_level = new_zoom

        # Compute where img_pt_before renders under new_zoom
        widget_pt_after = self.image_to_widget_pt(img_pt_before)

        # Shift pan_offset so image point stays exactly under cursor
        self.pan_offset += (cursor_widget_pt - widget_pt_after)

        self.clamp_pan_offset()
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.img_pixmap:
            img_pt = self.widget_to_image_coords(event.position())
            iw, ih = self.img_size.width(), self.img_size.height()

            if 0 <= img_pt.x() <= iw and 0 <= img_pt.y() <= ih:
                clicked_idx = -1
                for idx, b in enumerate(self.boxes):
                    bw = b['width'] * iw
                    bh = b['height'] * ih
                    bx = (b['x_center'] * iw) - (bw / 2.0)
                    by = (b['y_center'] * ih) - (bh / 2.0)
                    box_rect = QRectF(bx, by, bw, bh)
                    if box_rect.contains(img_pt):
                        clicked_idx = idx

                if clicked_idx != -1:
                    self.selected_box_idx = clicked_idx
                    self.box_changed.emit()
                    self.update()
                elif self.tool_mode == 'select':
                    self.selected_box_idx = -1
                    self.box_changed.emit()
                    self.update()

                if self.tool_mode == 'draw':
                    self.is_drawing = True
                    self.draw_start_img = img_pt
                    self.current_draw_rect_img = QRectF(img_pt.x(), img_pt.y(), 0.0, 0.0)

            event.accept()

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.position() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.position()
            self.clamp_pan_offset()
            self.update()
            event.accept()
            return

        if self.img_pixmap:
            img_pt = self.widget_to_image_coords(event.position())
            self.mouse_img_pos = img_pt

            if self.is_drawing and self.tool_mode == 'draw':
                x1 = min(self.draw_start_img.x(), img_pt.x())
                y1 = min(self.draw_start_img.y(), img_pt.y())
                w = abs(img_pt.x() - self.draw_start_img.x())
                h = abs(img_pt.y() - self.draw_start_img.y())
                self.current_draw_rect_img = QRectF(x1, y1, w, h)

            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self.tool_mode == 'draw' else Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            r = self.current_draw_rect_img
            iw, ih = self.img_size.width(), self.img_size.height()

            if r.width() > 8 and r.height() > 8 and iw > 0 and ih > 0:
                xc = (r.x() + r.width() / 2.0) / iw
                yc = (r.y() + r.height() / 2.0) / ih
                bw = r.width() / iw
                bh = r.height() / ih

                new_box = {
                    "cls_id": self.selected_class_id,
                    "x_center": max(0.0, min(1.0, xc)),
                    "y_center": max(0.0, min(1.0, yc)),
                    "width": max(0.0, min(1.0, bw)),
                    "height": max(0.0, min(1.0, bh))
                }
                next_boxes = list(self.boxes)
                next_boxes.append(new_box)
                self.push_history(next_boxes)
                self.selected_box_idx = len(next_boxes) - 1

            self.current_draw_rect_img = QRectF()
            self.update()
            event.accept()

    def delete_selected_box(self):
        if 0 <= self.selected_box_idx < len(self.boxes):
            next_boxes = [b for i, b in enumerate(self.boxes) if i != self.selected_box_idx]
            self.push_history(next_boxes)
            self.selected_box_idx = -1
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor("#06080E"))

        if not self.img_pixmap or self.img_size.isEmpty():
            painter.setPen(QPen(QColor("#64748B")))
            painter.setFont(QFont("Inter, sans-serif", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded.\nSelect a valid dataset directory.")
            return

        iw = self.img_size.width()
        ih = self.img_size.height()

        full_img_rect = QRectF(0, 0, iw, ih)
        widget_img_rect = self.image_to_widget_rect(full_img_rect)

        painter.drawPixmap(widget_img_rect.toRect(), self.img_pixmap)

        for idx, b in enumerate(self.boxes):
            is_sel = (idx == self.selected_box_idx)
            cls_id = b.get('cls_id', 0)
            cls_info = next((c for c in CLASSES if c['id'] == cls_id), CLASSES[0])
            base_color = cls_info['color']

            bw = b['width'] * iw
            bh = b['height'] * ih
            bx = (b['x_center'] * iw) - (bw / 2.0)
            by = (b['y_center'] * ih) - (bh / 2.0)
            box_img_rect = QRectF(bx, by, bw, bh)

            w_box_rect = self.image_to_widget_rect(box_img_rect)

            pen_color = QColor("#FFFFFF") if is_sel else base_color
            pen_width = 3.5 if is_sel else 2.0

            painter.setPen(QPen(pen_color, pen_width))
            fill_color = QColor(base_color)
            fill_color.setAlpha(55 if is_sel else 30)
            painter.setBrush(QBrush(fill_color))
            painter.drawRect(w_box_rect)

            badge_h = 18
            badge_w = 70
            badge_y = w_box_rect.y() - badge_h if w_box_rect.y() > badge_h + 5 else w_box_rect.y()
            badge_rect = QRectF(w_box_rect.x(), badge_y, badge_w, badge_h)

            painter.fillRect(badge_rect, base_color)
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.setFont(QFont("Inter, sans-serif", 9, QFont.Weight.Bold))
            label_txt = f"★ {cls_info['name']}" if is_sel else cls_info['name']
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, label_txt)

        if not self.current_draw_rect_img.isEmpty():
            w_draw_rect = self.image_to_widget_rect(self.current_draw_rect_img)
            cls_info = next((c for c in CLASSES if c['id'] == self.selected_class_id), CLASSES[0])

            painter.setPen(QPen(QColor("#000000"), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(w_draw_rect)

            pen = QPen(cls_info['color'], 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(w_draw_rect)

        if widget_img_rect.contains(self.mouse_to_widget_pt(self.mouse_img_pos)):
            w_pt = self.mouse_to_widget_pt(self.mouse_img_pos)
            painter.setPen(QPen(QColor("#000000"), 1.2, Qt.PenStyle.SolidLine))
            painter.drawLine(int(w_pt.x()), int(widget_img_rect.top()), int(w_pt.x()), int(widget_img_rect.bottom()))
            painter.drawLine(int(widget_img_rect.left()), int(w_pt.y()), int(widget_img_rect.right()), int(w_pt.y()))

    def mouse_to_widget_pt(self, img_pt: QPointF) -> QPointF:
        r = self.image_to_widget_rect(QRectF(img_pt.x(), img_pt.y(), 0, 0))
        return r.topLeft()


class LabelinPyQt6App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labelin Desktop Suite - High Performance YOLO Annotator")
        self.resize(1440, 900)

        self.root_dir = os.getcwd()
        self.image_files = []
        self.current_idx = 0
        self.current_folder = "dataset_labeled"
        self.canvas = None

        self.init_ui()
        self.apply_dark_stylesheet()
        self.load_directory("dataset_labeled")

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Sleek, Compact Top Control Bar (Fixed Height: 54px)
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("TopToolbar")
        toolbar_frame.setFixedHeight(54)
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(12, 6, 12, 6)
        toolbar_layout.setSpacing(8)

        # Brand Badge
        brand_lbl = QLabel("🏷️ Labelin Pro")
        brand_lbl.setStyleSheet("font-weight: 800; font-size: 14px; color: #38BDF8; font-family: sans-serif; margin-right: 6px;")
        toolbar_layout.addWidget(brand_lbl)

        # Folder Selector Dropdown
        self.folder_combo = QComboBox()
        self.folder_combo.addItems(["Labeled Images (dataset_labeled)", "Unlabeled Images (dataset_raw)", "Custom Folder..."])
        self.folder_combo.setFixedWidth(230)
        self.folder_combo.currentIndexChanged.connect(self.on_folder_changed)
        toolbar_layout.addWidget(self.folder_combo)

        # Sync / Refresh Button
        self.sync_btn = QPushButton("🔄 Sync")
        self.sync_btn.clicked.connect(self.reload_current_directory)
        toolbar_layout.addWidget(self.sync_btn)

        # Auto-Label AI Button
        self.autolabel_btn = QPushButton("⚡ Auto-Label AI")
        self.autolabel_btn.setStyleSheet("background: #8B5CF6; color: #FFF; font-weight: bold; padding: 5px 12px; border-radius: 5px;")
        self.autolabel_btn.clicked.connect(self.open_autolabel_dialog)
        toolbar_layout.addWidget(self.autolabel_btn)

        # Separator line
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #334155;")
        toolbar_layout.addWidget(sep1)

        # Navigation Buttons & Counter
        self.prev_btn = QPushButton("◀ Prev (A)")
        self.prev_btn.clicked.connect(self.prev_image)
        toolbar_layout.addWidget(self.prev_btn)

        self.counter_lbl = QLabel("0 / 0")
        self.counter_lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #94A3B8; font-size: 12px; padding: 0 4px;")
        toolbar_layout.addWidget(self.counter_lbl)

        self.next_btn = QPushButton("Next (D) ▶")
        self.next_btn.clicked.connect(self.next_image)
        toolbar_layout.addWidget(self.next_btn)

        # Filename Display
        self.filename_lbl = QLabel("No Image Loaded")
        self.filename_lbl.setStyleSheet("font-weight: bold; color: #F8FAFC; font-size: 12px; font-family: monospace; padding: 0 6px;")
        toolbar_layout.addWidget(self.filename_lbl)

        toolbar_layout.addStretch()

        # Class Selection Badges (1: Car, 2: Moto, 3: Bus, 4: Truck)
        self.class_btns = []
        for cls in CLASSES:
            btn = QPushButton(f"{cls['id']+1}. {cls['name']}")
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #1E293B; border: 1px solid #334155; color: #FFF; font-weight: bold; padding: 4px 10px; border-radius: 5px; font-size: 12px;
                }}
                QPushButton:checked {{
                    background: {cls['color'].name()}40; border: 2px solid {cls['color'].name()}; color: #FFF;
                }}
            """)
            btn.clicked.connect(lambda checked, c_id=cls['id']: self.select_class(c_id))
            toolbar_layout.addWidget(btn)
            self.class_btns.append(btn)

        # Separator line
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #334155;")
        toolbar_layout.addWidget(sep2)

        # Tools & Zoom Reset
        self.draw_btn = QPushButton("✏ Draw")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setChecked(True)
        self.draw_btn.clicked.connect(lambda: self.set_tool_mode('draw'))
        toolbar_layout.addWidget(self.draw_btn)

        self.select_btn = QPushButton("↖ Select")
        self.select_btn.setCheckable(True)
        self.select_btn.clicked.connect(lambda: self.set_tool_mode('select'))
        toolbar_layout.addWidget(self.select_btn)

        self.reset_zoom_btn = QPushButton("🔍 100%")
        self.reset_zoom_btn.clicked.connect(lambda: self.canvas.reset_view() if self.canvas else None)
        toolbar_layout.addWidget(self.reset_zoom_btn)

        # Undo / Redo / Save Buttons
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.setFixedWidth(36)
        self.undo_btn.clicked.connect(lambda: self.canvas.undo() if self.canvas else None)
        toolbar_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("↷")
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.setFixedWidth(36)
        self.redo_btn.clicked.connect(lambda: self.canvas.redo() if self.canvas else None)
        toolbar_layout.addWidget(self.redo_btn)

        self.save_btn = QPushButton("💾 Save (S)")
        self.save_btn.setStyleSheet("background: #10B981; font-weight: bold; color: #FFF; padding: 5px 14px; border-radius: 6px; font-size: 12px;")
        self.save_btn.clicked.connect(self.save_current_annotations)
        toolbar_layout.addWidget(self.save_btn)

        # Add top bar to main layout with stretch factor 0
        main_layout.addWidget(toolbar_frame, 0)

        # Main Splitter Workspace (Canvas Left 82%, Sidebar Right 18%) - Takes 100% remaining vertical height
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")

        # Canvas Instantiation
        self.canvas = NativeAnnotationCanvas()
        self.canvas.box_changed.connect(self.update_sidebar_list)
        splitter.addWidget(self.canvas)

        # Right Drawer Sidebar
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("SidebarFrame")
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)

        self.sidebar_title = QLabel("Bounding Boxes (0)")
        self.sidebar_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFF;")
        sidebar_layout.addWidget(self.sidebar_title)

        self.box_list_widget = QListWidget()
        self.box_list_widget.itemClicked.connect(self.on_box_item_clicked)
        sidebar_layout.addWidget(self.box_list_widget)

        self.delete_btn = QPushButton("🗑 Delete Box (Space)")
        self.delete_btn.setStyleSheet("background: #EF4444; color: #FFF; font-weight: bold; padding: 8px; border-radius: 6px; font-size: 12px;")
        self.delete_btn.clicked.connect(lambda: self.canvas.delete_selected_box())
        sidebar_layout.addWidget(self.delete_btn)

        # Instructions / Shortcuts Panel
        shortcuts_info = QLabel(
            "<b>Shortcuts:</b><br/>"
            "• <b>Right-Click Drag</b>: Pan Canvas<br/>"
            "• <b>Mouse Wheel</b>: Zoom to Cursor<br/>"
            "• <b>A / D</b>: Prev / Next Image<br/>"
            "• <b>S / Ctrl+S</b>: Save<br/>"
            "• <b>Space / Del</b>: Delete Box<br/>"
            "• <b>1-4</b>: Switch Class"
        )
        shortcuts_info.setStyleSheet("font-size: 11px; color: #94A3B8; background: #06080E; padding: 10px; border-radius: 6px; border: 1px solid #1E293B;")
        sidebar_layout.addWidget(shortcuts_info)

        splitter.addWidget(sidebar_frame)
        splitter.setSizes([1150, 290])

        # Add splitter to main layout with stretch factor 1
        main_layout.addWidget(splitter, 1)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Labelin Desktop App Ready.")

        # Select initial class after canvas is ready
        self.select_class(0)

        # Setup Keyboard Shortcuts
        QShortcut(QKeySequence("A"), self, self.prev_image)
        QShortcut(QKeySequence("D"), self, self.next_image)
        QShortcut(QKeySequence("S"), self, self.save_current_annotations)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_annotations)
        QShortcut(QKeySequence("Ctrl+Z"), self, lambda: self.canvas.undo() if self.canvas else None)
        QShortcut(QKeySequence("Ctrl+Y"), self, lambda: self.canvas.redo() if self.canvas else None)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, lambda: self.canvas.redo() if self.canvas else None)
        QShortcut(QKeySequence("Delete"), self, lambda: self.canvas.delete_selected_box())
        QShortcut(QKeySequence("Space"), self, lambda: self.canvas.delete_selected_box())
        QShortcut(QKeySequence("Backspace"), self, lambda: self.canvas.delete_selected_box())
        QShortcut(QKeySequence("1"), self, lambda: self.select_class(0))
        QShortcut(QKeySequence("2"), self, lambda: self.select_class(1))
        QShortcut(QKeySequence("3"), self, lambda: self.select_class(2))
        QShortcut(QKeySequence("4"), self, lambda: self.select_class(3))

    def open_autolabel_dialog(self):
        dlg = AutoLabelDialog(self)
        dlg.exec()

    def apply_dark_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #090D16;
                color: #F8FAFC;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
            }
            #TopToolbar {
                background-color: #111827;
                border: 1px solid #1F2937;
                border-radius: 8px;
            }
            #SidebarFrame {
                background-color: #111827;
                border: 1px solid #1F2937;
                border-radius: 8px;
            }
            QComboBox, QPushButton {
                background-color: #1F2937;
                color: #F8FAFC;
                border: 1px solid #374151;
                padding: 4px 10px;
                border-radius: 5px;
                font-weight: 600;
                font-size: 12px;
            }
            QComboBox:hover, QPushButton:hover {
                background-color: #374151;
                border-color: #4B5563;
            }
            QListWidget {
                background-color: #06080E;
                border: 1px solid #1F2937;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 4px;
                margin-bottom: 3px;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background-color: #3B82F635;
                border: 1px solid #3B82F6;
                color: #FFF;
            }
            QStatusBar {
                background: #06080E;
                color: #64748B;
                font-size: 11px;
            }
        """)

    def on_folder_changed(self, index):
        if index == 0:
            self.load_directory("dataset_labeled")
        elif index == 1:
            self.load_directory("dataset_raw")
        elif index == 2:
            custom_dir = QFileDialog.getExistingDirectory(self, "Select Image Directory", self.root_dir)
            if custom_dir:
                self.load_directory(custom_dir)

    def reload_current_directory(self):
        self.load_directory(self.current_folder)

    def load_directory(self, folder_name_or_path):
        self.current_folder = folder_name_or_path

        if os.path.isabs(folder_name_or_path):
            img_dir = folder_name_or_path
        elif folder_name_or_path == "dataset_labeled":
            img_dir = os.path.join(self.root_dir, "dataset_labeled", "images")
            if not os.path.exists(img_dir):
                img_dir = os.path.join(self.root_dir, "dataset_labeled")
        elif folder_name_or_path == "dataset_raw":
            img_dir = os.path.join(self.root_dir, "dataset_raw")
        else:
            img_dir = folder_name_or_path

        if not os.path.exists(img_dir):
            self.image_files = []
        else:
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            def sort_key(name):
                try:
                    clean = name.replace("frame_", "").split('.')[0]
                    return int(clean)
                except ValueError:
                    return name
            files.sort(key=sort_key)
            self.image_files = [os.path.join(img_dir, f) for f in files]

        self.current_idx = 0
        self.display_current_image()

    def display_current_image(self):
        if not self.image_files:
            if self.canvas:
                self.canvas.load_image("")
            self.counter_lbl.setText("0 / 0")
            self.filename_lbl.setText("No Images Found")
            self.box_list_widget.clear()
            self.sidebar_title.setText("Bounding Boxes (0)")
            return

        self.current_idx = max(0, min(self.current_idx, len(self.image_files) - 1))
        img_path = self.image_files[self.current_idx]
        filename = os.path.basename(img_path)

        if self.canvas:
            self.canvas.load_image(img_path)
        self.counter_lbl.setText(f"{self.current_idx + 1} / {len(self.image_files)}")
        self.filename_lbl.setText(filename)

        boxes = self.read_labels_for_file(filename)
        if self.canvas:
            self.canvas.set_boxes(boxes)
        self.update_sidebar_list()
        self.statusBar.showMessage(f"Loaded {filename} ({len(boxes)} bounding boxes).")

    def read_labels_for_file(self, filename):
        stem = filename.rsplit('.', 1)[0] + '.txt'
        manual_label_path = os.path.join(self.root_dir, "dataset_manual", "labels", stem)
        labeled_label_path = os.path.join(self.root_dir, "dataset_labeled", "labels", stem)

        label_path = None
        if os.path.exists(manual_label_path) and os.path.getsize(manual_label_path) > 0:
            label_path = manual_label_path
        elif os.path.exists(labeled_label_path) and os.path.getsize(labeled_label_path) > 0:
            label_path = labeled_label_path

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
        return boxes

    def save_current_annotations(self):
        if not self.image_files or self.current_idx >= len(self.image_files):
            return

        img_path = self.image_files[self.current_idx]
        filename = os.path.basename(img_path)
        stem = filename.rsplit('.', 1)[0] + '.txt'

        manual_labels_dir = os.path.join(self.root_dir, "dataset_manual", "labels")
        manual_images_dir = os.path.join(self.root_dir, "dataset_manual", "images")
        os.makedirs(manual_labels_dir, exist_ok=True)
        os.makedirs(manual_images_dir, exist_ok=True)

        target_label_path = os.path.join(manual_labels_dir, stem)
        target_image_path = os.path.join(manual_images_dir, filename)

        boxes_to_save = self.canvas.boxes if self.canvas else []
        lines = []
        for b in boxes_to_save:
            lines.append(f"{b['cls_id']} {b['x_center']:.6f} {b['y_center']:.6f} {b['width']:.6f} {b['height']:.6f}")

        with open(target_label_path, 'w') as f:
            f.write('\n'.join(lines))

        if not os.path.exists(target_image_path) and os.path.exists(img_path):
            img_data = cv2.imread(img_path)
            if img_data is not None:
                cv2.imwrite(target_image_path, img_data)

        self.statusBar.showMessage(f"✓ Saved {len(boxes_to_save)} boxes to {stem}!")

    def prev_image(self):
        if self.image_files and self.current_idx > 0:
            self.save_current_annotations()
            self.current_idx -= 1
            self.display_current_image()

    def next_image(self):
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.save_current_annotations()
            self.current_idx += 1
            self.display_current_image()

    def select_class(self, cls_id):
        if self.canvas:
            self.canvas.selected_class_id = cls_id
        for btn in self.class_btns:
            btn.setChecked(False)
        if 0 <= cls_id < len(self.class_btns):
            self.class_btns[cls_id].setChecked(True)

    def set_tool_mode(self, mode):
        if self.canvas:
            self.canvas.tool_mode = mode
        self.draw_btn.setChecked(mode == 'draw')
        self.select_btn.setChecked(mode == 'select')

    def update_sidebar_list(self):
        self.box_list_widget.clear()
        boxes_list = self.canvas.boxes if self.canvas else []
        selected_idx = self.canvas.selected_box_idx if self.canvas else -1

        self.sidebar_title.setText(f"Bounding Boxes ({len(boxes_list)})")

        for idx, b in enumerate(boxes_list):
            cls_id = b.get('cls_id', 0)
            cls_info = next((c for c in CLASSES if c['id'] == cls_id), CLASSES[0])
            item = QListWidgetItem(f"[{cls_info['name']}]  xc:{b['x_center']:.2f} yc:{b['y_center']:.2f}")
            if idx == selected_idx:
                item.setSelected(True)
            self.box_list_widget.addItem(item)

    def on_box_item_clicked(self, item):
        row = self.box_list_widget.row(item)
        if self.canvas:
            self.canvas.selected_box_idx = row
            self.canvas.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LabelinPyQt6App()
    window.show()
    sys.exit(app.exec())
