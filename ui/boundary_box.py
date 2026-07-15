import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer

class BoundaryBox(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skribbl Capture Area")
        # Ensure it stays on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        # Make the window itself mostly transparent
        self.setWindowOpacity(0.3)
        self.setStyleSheet("background-color: red;")

        # Load initial geometry
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.bbox_file = os.path.join(base_dir, 'data', 'canvas_bbox.json')
        if os.path.exists(self.bbox_file):
            try:
                with open(self.bbox_file, "r") as f:
                    data = json.load(f)
                    self.setGeometry(data["x"], data["y"], data["width"], data["height"])
            except:
                self.setGeometry(100, 100, 800, 600)
        else:
            self.setGeometry(100, 100, 800, 600)
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.save_geometry)
        self.timer.start(500) # Save every 500ms
        
    def save_geometry(self):
        # We need the global coordinates of the inner drawable area
        top_left = self.mapToGlobal(self.rect().topLeft())
        data = {
            "x": top_left.x(),
            "y": top_left.y(),
            "width": self.width(),
            "height": self.height()
        }
        # Write to a temporary file, then rename for atomic update
        tmp_file = self.bbox_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            os.replace(tmp_file, self.bbox_file)
        except Exception as e:
            print(f"Error saving bbox: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BoundaryBox()
    window.show()
    sys.exit(app.exec())
