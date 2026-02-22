import sys
import os
import cv2
import shutil
import random
import glob
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QTabWidget, QSplitter, 
    QListWidget, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsPixmapItem, QMessageBox, QFileDialog, QFormLayout, 
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QProgressBar, 
    QGroupBox, QListWidgetItem, QGraphicsTextItem, QScrollArea, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QObject
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QFont, QBrush, QPainter

# Settings
def load_classes():
    if os.path.exists("classes.txt"):
        with open("classes.txt", "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
            if not classes:
                classes = ['car', 'motorcycle', 'bus', 'truck']
    else:
        classes = ['car', 'motorcycle', 'bus', 'truck']
        with open("classes.txt", "w") as f:
            f.write("\n".join(classes))
    return classes

CLASSES = load_classes()
COLORS_QT = [QColor(0, 255, 0), QColor(255, 0, 0), QColor(0, 0, 255), QColor(255, 255, 0)]
while len(COLORS_QT) < len(CLASSES):
    COLORS_QT.append(QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))

class StdOutRedirect(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass

# ==========================================
# THREADS FOR BACKGROUND TASKS
# ==========================================

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
                
        count = 0
        saved = 0
        self.log.emit(f"Starting capture from frame_{start_idx:04d}.jpg...")
        
        while saved < self.sb_num_frames.value() and caps:
            active_caps = []
            for src_name, cap in caps:
                ret, frame = cap.read()
                if ret:
                    active_caps.append((src_name, cap))
                    count += 1
                    if count % self.frame_skip == 0:
                        current_frame_id = start_idx + saved
                        filename = f"{self.output_dir}/frame_{current_frame_id:04d}.jpg"
                        cv2.imwrite(filename, frame)
                        saved += 1
                        self.progress.emit(saved, self.sb_num_frames.value())
                        self.log.emit(f"Saved frame {current_frame_id:04d} from {src_name} ({saved}/{self.sb_num_frames.value()})")
                        
                        if saved >= self.sb_num_frames.value():
                            break
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


# ==========================================
# CUSTOM GRAPHICS VIEW FOR LABELING
# ==========================================

class BoundingBoxItem(QGraphicsRectItem):
    def __init__(self, rect, cls_id, parent_view):
        super().__init__(rect)
        self.cls_id = cls_id
        self.parent_view = parent_view
        
        color = COLORS_QT[cls_id % len(COLORS_QT)]
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        
        class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
        self.text_item = QGraphicsTextItem(class_name, self)
        self.text_item.setDefaultTextColor(color)
        self.text_item.setPos(rect.topLeft())
        font = QFont()
        font.setBold(True)
        self.text_item.setFont(font)
        
        self.resizing = False
        self.resize_margin = 8
        self.current_edge = None

    def get_edge(self, pos):
        rect = self.rect()
        m = self.resize_margin
        left = abs(pos.x() - rect.left()) < m
        right = abs(pos.x() - rect.right()) < m
        top = abs(pos.y() - rect.top()) < m
        bottom = abs(pos.y() - rect.bottom()) < m

        if left and top: return 'top_left'
        if right and bottom: return 'bottom_right'
        if right and top: return 'top_right'
        if left and bottom: return 'bottom_left'
        if left: return 'left'
        if right: return 'right'
        if top: return 'top'
        if bottom: return 'bottom'
        return None

    def hoverMoveEvent(self, event):
        edge = self.get_edge(event.pos())
        if edge in ['top', 'bottom']:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif edge in ['left', 'right']:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif edge in ['top_left', 'bottom_right']:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edge in ['top_right', 'bottom_left']:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self.get_edge(event.pos())
            if edge:
                self.resizing = True
                self.current_edge = edge
                self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, False)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            pos = event.pos()
            rect = self.rect()
            
            if 'left' in self.current_edge:
                rect.setLeft(min(pos.x(), rect.right() - 5))
            elif 'right' in self.current_edge:
                rect.setRight(max(pos.x(), rect.left() + 5))
                
            if 'top' in self.current_edge:
                rect.setTop(min(pos.y(), rect.bottom() - 5))
            elif 'bottom' in self.current_edge:
                rect.setBottom(max(pos.y(), rect.top() + 5))
                
            self.setRect(rect)
            self.text_item.setPos(rect.topLeft())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.resizing:
            self.resizing = False
            self.current_edge = None
            self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
            if hasattr(self.parent_view, 'parent_tab'):
                self.parent_view.parent_tab.update_box_list()
        else:
            super().mouseReleaseEvent(event)

    def update_rect(self, start_pos, end_pos):
        x1, y1 = min(start_pos.x(), end_pos.x()), min(start_pos.y(), end_pos.y())
        x2, y2 = max(start_pos.x(), end_pos.x()), max(start_pos.y(), end_pos.y())
        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        self.setRect(rect)
        self.text_item.setPos(rect.topLeft())


class LabelCanvas(QGraphicsView):
    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.current_image_item = None
        self.drawing = False
        self.start_pos = None
        self.current_box = None
        
        self.boxes = []
        
    def load_image(self, img_path):
        self.scene.clear()
        self.boxes.clear()
        
        pixmap = QPixmap(img_path)
        if pixmap.isNull(): return False
        
        self.current_image_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return True
        
    def add_box(self, cls_id, rect):
        box_item = BoundingBoxItem(rect, cls_id, self)
        self.scene.addItem(box_item)
        self.boxes.append(box_item)
        return box_item
        
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, BoundingBoxItem) or (item and isinstance(item.parentItem(), BoundingBoxItem)):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton and self.current_image_item:
            self.drawing = True
            pos = self.mapToScene(event.pos())
            self.start_pos = pos
            cls_id = self.parent_tab.get_selected_class()
            self.current_box = BoundingBoxItem(QRectF(pos, pos), cls_id, self)
            self.scene.addItem(self.current_box)
            self.boxes.append(self.current_box)
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        if self.drawing and self.current_box:
            current_pos = self.mapToScene(event.pos())
            
            # constrain
            scene_rect = self.scene.sceneRect()
            x = max(0, min(current_pos.x(), scene_rect.width()))
            y = max(0, min(current_pos.y(), scene_rect.height()))
            constrained_pos = QPointF(x, y)
            
            self.current_box.update_rect(self.start_pos, constrained_pos)
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.current_box:
                rect = self.current_box.rect()
                if rect.width() < 5 or rect.height() < 5:
                    self.scene.removeItem(self.current_box)
                    self.boxes.remove(self.current_box)
            self.current_box = None
            self.parent_tab.update_box_list()
        super().mouseReleaseEvent(event)

    def delete_selected(self):
        items = self.scene.selectedItems()
        for item in items:
            if isinstance(item, BoundingBoxItem):
                self.scene.removeItem(item)
                if item in self.boxes:
                    self.boxes.remove(item)
        self.parent_tab.update_box_list()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_in_factor = 1.1
            zoom_out_factor = 1 / zoom_in_factor
            if event.angleDelta().y() > 0:
                zoom_factor = zoom_in_factor
            else:
                zoom_factor = zoom_out_factor
            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)


# ==========================================
# MAIN APPLICATION
# ==========================================

class TrafficYoloApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic YOLO Suite - Professional")
        self.resize(1200, 800)
        
        # apply style
        self.apply_dark_theme()
        
        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_step1_tab(), "1. Data Acquisition (Auto-Label)")
        self.tabs.addTab(self.create_step2_tab(), "2. Labeling Workspace")
        self.tabs.addTab(self.create_step3_tab(), "3. Dataset Prep")
        self.tabs.addTab(self.create_step4_tab(), "4. Model Training")
        
        main_layout.addWidget(self.tabs)
        
        # State vars UI
        self.dataset_raw = "dataset_raw"
        self.dataset_labeled = "dataset_labeled"
        self.dataset_manual = "dataset_manual" # Output of workspace
        
        self.current_images = []
        self.current_img_idx = 0
        
    def apply_dark_theme(self):
        style = """
        QMainWindow, QDialog, QMessageBox { background-color: #1e1e2e; }
        QWidget { color: #cdd6f4; font-family: 'Segoe UI', Arial; font-size: 14pt; }
        QTabWidget::pane { border: 1px solid #45475a; border-radius: 8px; background: #181825; }
        QTabBar::tab { background: #313244; padding: 12px 24px; border-radius: 6px; margin: 2px; font-weight: bold; }
        QTabBar::tab:selected { background: #89b4fa; color: #11111b; font-weight: bold; font-size: 15pt; }
        QPushButton { background-color: #89b4fa; color: #11111b; border: none; padding: 14px; border-radius: 6px; font-weight: bold; font-size: 13pt; }
        QPushButton:hover { background-color: #b4befe; }
        QPushButton:disabled { background-color: #45475a; color: #a6adc8; }
        QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 8px; font-size: 14pt; }
        QTextEdit { background-color: #11111b; color: #a6e3a1; font-family: Consolas, monospace; border: 1px solid #45475a; font-size: 12pt; padding: 10px; }
        QListWidget { background-color: #181825; border: 1px solid #45475a; border-radius: 6px; font-size: 14pt; padding: 5px; }
        QListWidget::item { padding: 8px; border-bottom: 1px solid #313244; }
        QListWidget::item:selected { background-color: #45475a; border-radius: 4px; }
        QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 25px; font-weight: bold; font-size: 15pt; padding-top: 15px; }
        QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; color: #89b4fa; top: -10px; }
        QProgressBar { text-align: center; border: 1px solid #45475a; border-radius: 6px; font-weight: bold; font-size: 12pt; height: 25px; }
        QProgressBar::chunk { background-color: #a6e3a1; border-radius: 6px; }
        QSplitter::handle { background-color: #45475a; }
        QTreeView, QListView, QTableView { background-color: #181825; border: 1px solid #45475a; }
        QHeaderView::section { background-color: #313244; border: 1px solid #45475a; padding: 4px; }
        """
        self.setStyleSheet(style)

    # ------------------ STEP 1: ACQUISITION ------------------ #
    def create_step1_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        
        # 1. Capture Group
        grp1 = QGroupBox("Capture Streams")
        l1 = QFormLayout()
        l1.setSpacing(15)
        
        self.source_list = QListWidget()
        self.source_list.setFixedHeight(120)
        self.source_list.addItem("http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8")
        
        btn_add_url = QPushButton("+ Add URL")
        btn_add_url.setStyleSheet("font-size: 11pt; padding: 5px;")
        btn_add_url.clicked.connect(self.add_stream_url)
        
        btn_add_vid = QPushButton("+ Add Video")
        btn_add_vid.setStyleSheet("font-size: 11pt; padding: 5px;")
        btn_add_vid.clicked.connect(self.add_stream_video)
        
        btn_del_src = QPushButton("- Remove")
        btn_del_src.setStyleSheet("font-size: 11pt; padding: 5px; background-color: #f38ba8; color: #11111b;")
        btn_del_src.clicked.connect(self.remove_stream_source)
        
        h_src_btns = QHBoxLayout()
        h_src_btns.addWidget(btn_add_url)
        h_src_btns.addWidget(btn_add_vid)
        h_src_btns.addWidget(btn_del_src)
        
        v_src = QVBoxLayout()
        v_src.addWidget(self.source_list)
        v_src.addLayout(h_src_btns)
        
        self.num_frames_in = QSpinBox()
        self.num_frames_in.setRange(10, 5000)
        self.num_frames_in.setValue(100)
        
        self.frame_skip_in = QSpinBox()
        self.frame_skip_in.setRange(1, 300)
        self.frame_skip_in.setValue(30)
        
        self.btn_capture = QPushButton("Start Capture")
        self.btn_capture.clicked.connect(self.start_capture)
        
        self.cap_prog = QProgressBar()
        
        l1.addRow("Video Sources:", v_src)
        l1.addRow("Frames to Capture:", self.num_frames_in)
        l1.addRow("Capture Every N Frames:", self.frame_skip_in)
        l1.addRow(self.btn_capture)
        l1.addRow(self.cap_prog)
        grp1.setLayout(l1)
        
        # 2. Auto-Labeling Group
        grp2 = QGroupBox("Auto-Labeling (YOLO Pre-annotate)")
        l2 = QFormLayout()
        l2.setSpacing(15)
        self.al_model_in = QLineEdit("../yolov26n.pt")
        btn_al_model = QPushButton("Browse")
        btn_al_model.setStyleSheet("font-size: 11pt; padding: 5px;")
        btn_al_model.clicked.connect(lambda: self.browse_file(self.al_model_in, "Select YOLO Model", "*.pt"))
        h_al = QHBoxLayout()
        h_al.addWidget(self.al_model_in)
        h_al.addWidget(btn_al_model)
        
        self.btn_autolabel = QPushButton("Start Auto-Label")
        self.btn_autolabel.clicked.connect(self.start_autolabel)
        
        self.al_prog = QProgressBar()
        
        l2.addRow("YOLO Model Path (.pt):", h_al)
        l2.addRow(self.btn_autolabel)
        l2.addRow(self.al_prog)
        grp2.setLayout(l2)
        
        # Log Box
        self.s1_log = QTextEdit()
        self.s1_log.setReadOnly(True)
        
        lay.addWidget(grp1)
        lay.addWidget(grp2)
        lbl_logs = QLabel("Logs:")
        lbl_logs.setStyleSheet("font-weight: bold; font-size: 15pt; color: #89b4fa;")
        lay.addWidget(lbl_logs)
        lay.addWidget(self.s1_log)
        
        return w

    def add_stream_url(self):
        url, ok = QInputDialog.getText(self, "Add Stream URL", "Enter RTSP/HTTP or Webcam ID (e.g. 0):")
        if ok and url.strip():
            self.source_list.addItem(url.strip())
            
    def add_stream_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv)", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.source_list.addItem(path)
            
    def remove_stream_source(self):
        row = self.source_list.currentRow()
        if row >= 0:
            self.source_list.takeItem(row)

    def log_s1(self, msg):
        self.s1_log.append(msg)

    def start_capture(self):
        sources = [self.source_list.item(i).text() for i in range(self.source_list.count())]
        if not sources:
            QMessageBox.warning(self, "Error", "Please add at least one video source.")
            return
            
        self.btn_capture.setEnabled(False)
        self.cap_prog.setValue(0)
        self.cap_thread = CaptureThread(
            sources, 
            self.num_frames_in, 
            self.frame_skip_in.value(),
            "dataset_raw"
        )
        self.cap_thread.progress.connect(lambda cur, tot: self.cap_prog.setValue(int(cur/tot*100)))
        self.cap_thread.log.connect(self.log_s1)
        self.cap_thread.finished.connect(self.on_capture_finished)
        self.cap_thread.start()

    def on_capture_finished(self, success, msg):
        self.btn_capture.setEnabled(True)
        self.log_s1(msg)
        if success:
            QMessageBox.information(self, "Capture", "Capture completed successfully.")

    def start_autolabel(self):
        self.btn_autolabel.setEnabled(False)
        self.al_prog.setValue(0)
        self.al_thread = AutoLabelThread(
            self.al_model_in.text(),
            "dataset_raw",
            "dataset_labeled"
        )
        self.al_thread.progress.connect(lambda cur, tot: self.al_prog.setValue(int(cur/tot*100)))
        self.al_thread.log.connect(self.log_s1)
        self.al_thread.finished.connect(self.on_autolabel_finished)
        self.al_thread.start()

    def on_autolabel_finished(self, success, msg):
        self.btn_autolabel.setEnabled(True)
        self.log_s1(msg)
        if success:
            QMessageBox.information(self, "Auto-Label", "Auto-labeling completed.")


    # ------------------ STEP 2: WORKSPACE ------------------ #
    def create_step2_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        
        split = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Tools & Classes
        left_panel = QWidget()
        lv = QVBoxLayout(left_panel)
        
        # Load Buttons
        btn_load_raw = QPushButton("Load Raw Images")
        btn_load_raw.clicked.connect(lambda: self.load_workspace("dataset_raw", is_labeled=False))
        
        btn_load_auto = QPushButton("Load Auto-Labeled")
        btn_load_auto.clicked.connect(lambda: self.load_workspace("dataset_labeled/images", is_labeled=True))
        
        lv.addWidget(btn_load_raw)
        lv.addWidget(btn_load_auto)
        
        # Class Selector
        lbl_classes = QLabel("Classes:")
        lbl_classes.setStyleSheet("font-weight: bold; font-size: 15pt; color: #89b4fa; margin-top: 10px;")
        
        btn_add_class = QPushButton("Add")
        btn_add_class.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-size: 11pt; padding: 5px;")
        btn_add_class.clicked.connect(self.add_class)
        
        btn_edit_class = QPushButton("Edit")
        btn_edit_class.setStyleSheet("background-color: #fab387; color: #11111b; font-size: 11pt; padding: 5px;")
        btn_edit_class.clicked.connect(self.edit_class)
        
        btn_del_class = QPushButton("Del")
        btn_del_class.setStyleSheet("background-color: #f38ba8; color: #11111b; font-size: 11pt; padding: 5px;")
        btn_del_class.clicked.connect(self.delete_class)
        
        ch_lay = QHBoxLayout()
        ch_lay.addWidget(lbl_classes)
        ch_lay.addWidget(btn_add_class)
        ch_lay.addWidget(btn_edit_class)
        ch_lay.addWidget(btn_del_class)
        
        lv.addLayout(ch_lay)
        
        self.class_list = QListWidget()
        for i, c in enumerate(CLASSES):
            item = QListWidgetItem(f"{i} : {c}")
            font = QFont()
            font.setBold(True)
            font.setPointSize(14)
            item.setFont(font)
            item.setForeground(COLORS_QT[i])
            self.class_list.addItem(item)
        self.class_list.setCurrentRow(0)
        lv.addWidget(self.class_list)
        
        # Box List
        lbl_boxes = QLabel("Current Boxes:")
        lbl_boxes.setStyleSheet("font-weight: bold; font-size: 15pt; color: #89b4fa; margin-top: 10px;")
        lv.addWidget(lbl_boxes)
        self.box_list = QListWidget()
        lv.addWidget(self.box_list)
        
        btn_del = QPushButton("Delete Selected Box")
        btn_del.setStyleSheet("background-color: #f38ba8; color: #11111b;")
        btn_del.clicked.connect(self.delete_selected_box)
        lv.addWidget(btn_del)
        
        split.addWidget(left_panel)
        
        # Middle Panel: Canvas
        mid_panel = QWidget()
        mv = QVBoxLayout(mid_panel)
        self.canvas = LabelCanvas(self)
        mv.addWidget(self.canvas)
        
        nav_lay = QHBoxLayout()
        nav_lay.setSpacing(20)
        self.btn_prev = QPushButton("◄ Prev Image (A)")
        self.btn_prev.clicked.connect(self.prev_image)
        self.lbl_img_info = QLabel("Image 0/0")
        self.lbl_img_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_info.setStyleSheet("font-weight: bold; font-size: 16pt; background: #313244; border-radius: 6px; padding: 10px;")
        self.btn_save = QPushButton("Save (S)")
        self.btn_save.setStyleSheet("background-color: #a6e3a1; color: #11111b;")
        self.btn_save.clicked.connect(self.save_current_labels)
        self.btn_next = QPushButton("Next Image (D) ►")
        self.btn_next.clicked.connect(self.next_image)
        
        nav_lay.addWidget(self.btn_prev)
        nav_lay.addWidget(self.lbl_img_info)
        nav_lay.addWidget(self.btn_save)
        nav_lay.addWidget(self.btn_next)
        mv.addLayout(nav_lay)
        
        split.addWidget(mid_panel)
        
        split.setSizes([200, 800])
        lay.addWidget(split)
        return w

    def browse_file(self, line_edit, title, filter_str="All Files (*)"):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str, options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            line_edit.setText(path)
            
    def browse_dir(self, line_edit, title):
        path = QFileDialog.getExistingDirectory(self, title, options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            line_edit.setText(path)

    def _sync_classes_file(self):
        with open("classes.txt", "w") as f:
            f.write("\n".join(CLASSES))
            
    def _refresh_class_list_ui(self):
        # Refresh the class list
        curr_row = self.class_list.currentRow()
        self.class_list.clear()
        
        # Ensure we have enough colors
        while len(COLORS_QT) < len(CLASSES):
            COLORS_QT.append(QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
            
        for i, c in enumerate(CLASSES):
            item = QListWidgetItem(f"{i} : {c}")
            font = QFont()
            font.setBold(True)
            font.setPointSize(14)
            item.setFont(font)
            item.setForeground(COLORS_QT[i])
            self.class_list.addItem(item)
            
        if curr_row >= len(CLASSES):
            curr_row = len(CLASSES) - 1
        self.class_list.setCurrentRow(max(0, curr_row))
        self.update_box_list()
        
    def add_class(self):
        text, ok = QInputDialog.getText(self, "Add Class", "New class name:")
        if ok and text.strip():
            CLASSES.append(text.strip())
            self._sync_classes_file()
            self._refresh_class_list_ui()

    def edit_class(self):
        row = self.class_list.currentRow()
        if row < 0 or row >= len(CLASSES): return
        
        old_name = CLASSES[row]
        text, ok = QInputDialog.getText(self, "Edit Class", "Ext. Class name:", text=old_name)
        if ok and text.strip() and text.strip() != old_name:
            CLASSES[row] = text.strip()
            self._sync_classes_file()
            self._refresh_class_list_ui()
            
            # update drawn boxes text
            for box in self.canvas.boxes:
                if box.cls_id == row:
                    box.text_item.setPlainText(text.strip())

    def delete_class(self):
        row = self.class_list.currentRow()
        if row < 0 or row >= len(CLASSES): return
        
        cls_name = CLASSES[row]
        reply = QMessageBox.question(self, "Confirm Delete", 
                                     f"Are you sure you want to delete class '{cls_name}' (ID: {row})?\n"
                                     "This will permanently shift subsequent class IDs and delete all boxes of this class!",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                                     
        if reply == QMessageBox.StandardButton.Yes:
            # 1. Remove from global list and update classes.txt
            del CLASSES[row]
            self._sync_classes_file()
            self._refresh_class_list_ui()
            
            # 2. Iterate and update all saved .txt labels in dataset_manual
            lbl_dir = os.path.join(self.dataset_manual, "labels")
            if os.path.exists(lbl_dir):
                for txt_file in os.listdir(lbl_dir):
                    if not txt_file.endswith('.txt'): continue
                    txt_path = os.path.join(lbl_dir, txt_file)
                    try:
                        with open(txt_path, 'r') as f:
                            lines = f.readlines()
                            
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if not parts: continue
                            cls_id = int(parts[0])
                            
                            if cls_id == row:
                                # This is the deleted class -> drop that line completely
                                continue
                            elif cls_id > row:
                                # This class index is shifted down by 1
                                parts[0] = str(cls_id - 1)
                                
                            new_lines.append(" ".join(parts))
                            
                        with open(txt_path, 'w') as f:
                            f.write("\n".join(new_lines))
                    except Exception as e:
                        print(f"Failed processing {txt_file}: {e}")
                        
            # 3. Update current canvas active drawn boxes
            boxes_to_remove = []
            for box in self.canvas.boxes:
                if box.cls_id == row:
                    boxes_to_remove.append(box)
                elif box.cls_id > row:
                    # Shift ID down by 1 in live session
                    box.cls_id -= 1
                    
            # Remove marked
            for b_rem in boxes_to_remove:
                self.canvas.scene.removeItem(b_rem)
                if b_rem in self.canvas.boxes:
                    self.canvas.boxes.remove(b_rem)
                    
            self.update_box_list()
            # If current active file isn't automatically saved: we'll call save anyway.
            self.save_current_labels()

    def get_selected_class(self):
        return self.class_list.currentRow()
        
    def update_box_list(self):
        self.box_list.clear()
        for i, b in enumerate(self.canvas.boxes):
            class_name = CLASSES[b.cls_id] if b.cls_id < len(CLASSES) else f"Class {b.cls_id}"
            item = QListWidgetItem(f"Box {i} - {class_name}")
            item.setData(Qt.ItemDataRole.UserRole, b)
            self.box_list.addItem(item)

    def delete_selected_box(self):
        self.canvas.delete_selected()

    def load_workspace(self, img_dir, is_labeled=False):
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "Error", f"Directory {img_dir} does not exist.")
            return
            
        self.current_images = [os.path.abspath(os.path.join(img_dir, f)) for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))]
        self.current_images.sort()
        self.current_img_idx = 0
        self.is_loading_autolabeled = is_labeled
        
        self.dataset_manual = "dataset_manual"
        os.makedirs(f"{self.dataset_manual}/images", exist_ok=True)
        os.makedirs(f"{self.dataset_manual}/labels", exist_ok=True)
        
        # Additional step to copy auto-labeled results as default manual labels upon load to enable editing
        if is_labeled:
            for img_name in os.listdir(img_dir):
                if img_name.endswith(('.jpg', '.png')):
                    lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                    auto_lbl_src = os.path.join(os.path.dirname(img_dir), "labels", lbl_name)
                    man_tgt = os.path.join(self.dataset_manual, "labels", lbl_name)
                    # Copy if manual target doesn't exist, OR if it exists but is completely empty (0 bytes)
                    should_copy = False
                    if not os.path.exists(man_tgt):
                        should_copy = True
                    elif os.path.getsize(man_tgt) == 0:
                        should_copy = True
                        
                    if os.path.exists(auto_lbl_src) and should_copy:
                        shutil.copy(auto_lbl_src, man_tgt)
                        
        self.load_current_image()

    def load_current_image(self):
        if not self.current_images:
            self.lbl_img_info.setText("No Images")
            self.canvas.scene.clear()
            return

        img_path = self.current_images[self.current_img_idx]
        
        # Strip any existing [SAVED] from base string just in case
        base_name = os.path.basename(img_path)
        self.lbl_img_info.setText(f"{base_name} ({self.current_img_idx+1}/{len(self.current_images)})")
        
        self.canvas.load_image(img_path)
        
        # Determine label path mapping
        # Try to find corresponding label (if auto-labeled or previously manual-labeled)
        base_name = os.path.basename(img_path)
        label_name = base_name.replace('.jpg','.txt').replace('.png','.txt')
        
        # Check manual output first, but ensure it is not an empty saved state
        manual_txt = os.path.join(self.dataset_manual, "labels", label_name)
        
        if os.path.exists(manual_txt) and os.path.getsize(manual_txt) > 0:
            self.render_labels(manual_txt)
        elif self.is_loading_autolabeled:
            # check the autolabeled txt
            auto_txt = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", label_name)
            if os.path.exists(auto_txt):
                self.render_labels(auto_txt)

    def render_labels(self, txt_path):
        img_item = self.canvas.current_image_item
        if not img_item: return
        w = img_item.pixmap().width()
        h = img_item.pixmap().height()
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, nw, nh = map(float, parts[1:5])
                
                # unnormalize
                x1 = (xc - nw/2) * w
                y1 = (yc - nh/2) * h
                bw = nw * w
                bh = nh * h
                
                self.canvas.add_box(cls_id, QRectF(x1, y1, bw, bh))
                
        self.update_box_list()

    def save_current_labels(self):
        if not self.current_images: return
        
        img_path = self.current_images[self.current_img_idx]
        base_name = os.path.basename(img_path)
        dst_img = os.path.join(self.dataset_manual, "images", base_name)
        
        # copy image
        shutil.copy(img_path, dst_img)
        
        # save labels
        label_name = base_name.replace('.jpg','.txt').replace('.png','.txt')
        dst_txt = os.path.join(self.dataset_manual, "labels", label_name)
        
        img_item = self.canvas.current_image_item
        w = img_item.pixmap().width()
        h = img_item.pixmap().height()
        
        lines = []
        for box in self.canvas.boxes:
            rect = box.rect()
            x1, y1 = rect.x(), rect.y()
            bw, bh = rect.width(), rect.height()
            
            # normalize
            xc = (x1 + bw/2) / w
            yc = (y1 + bh/2) / h
            nw = bw / w
            nh = bh / h
            
            lines.append(f"{box.cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            
        with open(dst_txt, 'w') as f:
            f.write("\n".join(lines))
        
        # Visual feedback
        curr_text = self.lbl_img_info.text()
        if not curr_text.endswith(" [SAVED]"):
            self.lbl_img_info.setText(curr_text + " [SAVED]")

    def next_image(self):
        if not self.current_images: return
        if self.current_img_idx < len(self.current_images) - 1:
            self.current_img_idx += 1
            self.load_current_image()

    def prev_image(self):
        if not self.current_images: return
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.load_current_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_A:
            self.save_current_labels()
            self.prev_image()
        elif event.key() == Qt.Key.Key_D:
            self.save_current_labels()
            self.next_image()
        elif event.key() == Qt.Key.Key_S:
            self.save_current_labels()
        elif event.key() in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace, Qt.Key.Key_Space]:
            self.delete_selected_box()
        elif event.key() == Qt.Key.Key_W:
            self.canvas.drawing = True 
            
        # Class Selection via Number Keys (0-9)
        elif Qt.Key.Key_0 <= event.key() <= Qt.Key.Key_9:
            num = event.key() - Qt.Key.Key_0
            if num < len(CLASSES):
                self.class_list.setCurrentRow(num)
                
        super().keyPressEvent(event)


    # ------------------ STEP 3: PREP DATASET ------------------ #
    def create_step3_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        
        grp = QGroupBox("Dataset Splitting Configuration")
        fl = QFormLayout()
        fl.setSpacing(15)
        
        self.src_dataset_in = QLineEdit(os.path.abspath("dataset_manual"))
        btn_src_dir = QPushButton("Browse")
        btn_src_dir.setStyleSheet("font-size: 11pt; padding: 5px;")
        btn_src_dir.clicked.connect(lambda: self.browse_dir(self.src_dataset_in, "Select Source Dataset Folder"))
        h_src = QHBoxLayout()
        h_src.addWidget(self.src_dataset_in)
        h_src.addWidget(btn_src_dir)
        
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.1, 0.9)
        self.train_ratio.setValue(0.8)
        self.train_ratio.setSingleStep(0.1)
        
        self.btn_split = QPushButton("Generate Train/Val Split & traffic.yaml")
        self.btn_split.clicked.connect(self.split_dataset)
        
        fl.addRow("Source Directory:", h_src)
        fl.addRow("Train Ratio (e.g. 0.8 = 80%):", self.train_ratio)
        fl.addRow(self.btn_split)
        
        grp.setLayout(fl)
        
        self.s3_log = QTextEdit()
        self.s3_log.setReadOnly(True)
        
        lay.addWidget(grp)
        lbl_logs3 = QLabel("Logs:")
        lbl_logs3.setStyleSheet("font-weight: bold; font-size: 15pt; color: #89b4fa;")
        lay.addWidget(lbl_logs3)
        lay.addWidget(self.s3_log)
        
        return w

    def split_dataset(self):
        source_dir = self.src_dataset_in.text()
        if not os.path.exists(source_dir):
            self.s3_log.append(f"Error: Directory {source_dir} not found. Please label images first.")
            return
            
        train_ratio = self.train_ratio.value()
        images = [f for f in os.listdir(os.path.join(source_dir, "images")) if f.endswith(('.jpg', '.png'))]
        if not images:
            self.s3_log.append("Error: No images found to split.")
            return
            
        self.btn_split.setEnabled(False)
        self.s3_log.append(f"Starting split... Total images: {len(images)}")
        
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # create structure
        base_dest = "dataset"
        for split in ['train', 'val']:
            os.makedirs(f"{base_dest}/{split}/images", exist_ok=True)
            os.makedirs(f"{base_dest}/{split}/labels", exist_ok=True)
            
        # Copy wrapper
        def copy_files(img_list, split_folder):
            for img in img_list:
                shutil.copy(os.path.join(source_dir, "images", img), os.path.join(base_dest, split_folder, "images"))
                label = img.replace('.jpg', '.txt').replace('.png', '.txt')
                l_path = os.path.join(source_dir, "labels", label)
                if os.path.exists(l_path):
                    shutil.copy(l_path, os.path.join(base_dest, split_folder, "labels"))

        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
        
        yaml_path = os.path.abspath(f"{base_dest}/traffic.yaml")
        yaml_content = f"""path: {os.path.abspath('dataset')}
train: train/images
val: val/images

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        self.s3_log.append(f"Success! {len(train_images)} train, {len(val_images)} val.")
        self.s3_log.append(f"YAML generated at: {yaml_path}")
        self.btn_split.setEnabled(True)

    # ------------------ STEP 4: TRAINING ------------------ #
    def create_step4_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        
        grp = QGroupBox("YOLO Training")
        fl = QFormLayout()
        fl.setSpacing(15)
        
        self.t_yaml = QLineEdit(os.path.abspath("dataset/traffic.yaml"))
        btn_t_yaml = QPushButton("Browse")
        btn_t_yaml.setStyleSheet("font-size: 11pt; padding: 5px;")
        btn_t_yaml.clicked.connect(lambda: self.browse_file(self.t_yaml, "Select traffic.yaml", "*.yaml"))
        h_yaml = QHBoxLayout()
        h_yaml.addWidget(self.t_yaml)
        h_yaml.addWidget(btn_t_yaml)
        
        self.t_model = QLineEdit("../yolov26n.pt")
        btn_t_model = QPushButton("Browse")
        btn_t_model.setStyleSheet("font-size: 11pt; padding: 5px;")
        btn_t_model.clicked.connect(lambda: self.browse_file(self.t_model, "Select Model .pt", "*.pt"))
        h_t_model = QHBoxLayout()
        h_t_model.addWidget(self.t_model)
        h_t_model.addWidget(btn_t_model)
        
        self.t_epochs = QSpinBox()
        self.t_epochs.setRange(1, 1000)
        self.t_epochs.setValue(50)
        
        self.t_batch = QSpinBox()
        self.t_batch.setRange(1, 128)
        self.t_batch.setValue(4)  # Default lowered to 4 for 4GB GPUs
        
        self.t_workers = QSpinBox()
        self.t_workers.setRange(0, 32)
        self.t_workers.setValue(0) # 0 to save RAM/VRAM mapping overhead
        
        self.t_imgsz = QSpinBox()
        self.t_imgsz.setRange(320, 1280)
        self.t_imgsz.setValue(640)
        self.t_imgsz.setSingleStep(32)
        
        self.btn_train = QPushButton("Start YOLO Training")
        self.btn_train.clicked.connect(self.start_training)
        
        fl.addRow("Traffic YAML Path:", h_yaml)
        fl.addRow("Base Model (.pt):", h_t_model)
        fl.addRow("Epochs:", self.t_epochs)
        fl.addRow("Batch Size:", self.t_batch)
        fl.addRow("Workers (CPU):", self.t_workers)
        fl.addRow("Image Size:", self.t_imgsz)
        fl.addRow(self.btn_train)
        grp.setLayout(fl)
        
        self.s4_log = QTextEdit()
        self.s4_log.setReadOnly(True)
        # Apply special dark terminal style for logs
        self.s4_log.setStyleSheet("background-color: #1e1e2e; color: #a6e3a1; font-family: monospace;")
        
        lay.addWidget(grp)
        lbl_logs4 = QLabel("Terminal Output:")
        lbl_logs4.setStyleSheet("font-weight: bold; font-size: 15pt; color: #89b4fa;")
        lay.addWidget(lbl_logs4)
        lay.addWidget(self.s4_log)
        
        return w

    def start_training(self):
        yaml_path = self.t_yaml.text()
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "Error", f"YAML not found: {yaml_path}")
            return
            
        # Redirect standard output / error
        self.stdout_redir = StdOutRedirect()
        self.stdout_redir.textWritten.connect(lambda t: self.s4_log.insertPlainText(t))
        sys.stdout = self.stdout_redir
        sys.stderr = self.stdout_redir
        
        self.btn_train.setEnabled(False)
        self.s4_log.append("--- TRAINING STARTED ---")
        
        self.train_thread = TrainThread(
            yaml_path,
            self.t_model.text() if os.path.exists(self.t_model.text()) else "yolo11n.pt",
            self.t_epochs.value(),
            self.t_batch.value(),
            self.t_imgsz.value(),
            self.t_workers.value()
        )
        self.train_thread.log.connect(lambda t: self.s4_log.append(t))
        self.train_thread.finished.connect(self.on_train_finished)
        self.train_thread.start()

    def on_train_finished(self, success, msg):
        self.btn_train.setEnabled(True)
        self.s4_log.append("\n" + msg)
        self.s4_log.append("--- TRAINING ENDED ---")
        # Restore sys out
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficYoloApp()
    window.show()
    sys.exit(app.exec())
