import os
import random
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor

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
