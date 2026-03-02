from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QMessageBox, QGraphicsTextItem
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont, QBrush, QPainter

from labelin.utils import CLASSES, COLORS_QT

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
