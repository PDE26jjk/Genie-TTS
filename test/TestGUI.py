import sys
import os

os.environ['GENIE_DATA_DIR'] = r"C:\Users\Haruka\Desktop\Midori\Data\common_resource\tts"

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from genie_tts.GUI.GUI import MainWindow

app = QApplication(sys.argv)
font = QFont("Microsoft YaHei", 10)
app.setFont(font)
window = MainWindow()
window.show()
sys.exit(app.exec())
