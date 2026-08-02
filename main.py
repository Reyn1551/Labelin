#!/usr/bin/env python3
"""
Labelin Suite - Native PyQt6 Desktop Application Entrypoint
Run this script to launch the high-performance native PyQt6 Traffic Dataset Annotator.
Usage:
    ./venv/bin/python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from labeling_app import LabelinPyQt6App

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LabelinPyQt6App()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
