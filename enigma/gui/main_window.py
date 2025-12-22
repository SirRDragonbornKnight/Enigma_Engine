"""
PyQt5 GUI for Enigma: Chat tab, Logbook (saved conversations), Training placeholder, Avatar control.
Saves conversations to enigma/data/conversations/.
Requires PyQt5 (already in requirements).
"""
import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QLabel, QListWidget, QTabWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ..core.inference import EnigmaEngine
from ..memory.manager import ConversationManager
from ..voice import speak, listen as transcribe_from_mic
from ..avatar.avatar_api import AvatarController
from ..config import CONFIG
import time

class STTThread(QThread):
    result = pyqtSignal(str)
    def __init__(self, timeout=8):
        super().__init__()
        self.timeout = timeout
    def run(self):
        text = transcribe_from_mic(timeout=self.timeout)
        self.result.emit(text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enigma — GUI")
        self.resize(900, 600)
        self.engine = EnigmaEngine()
        self.conv_manager = ConversationManager()
        self.avatar = AvatarController()
        self.current_messages = []
        self._build_ui()

    def _build_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "Chat")
        tabs.addTab(self._logbook_tab(), "Logbook")
        tabs.addTab(self._training_tab(), "Training")
        tabs.addTab(self._avatar_tab(), "Avatar")
        self.setCentralWidget(tabs)

    # Chat Tab
    def _chat_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        self.chat_display = QTextEdit(readOnly=True)
        self.chat_input = QLineEdit()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        self.tts_btn = QPushButton("Speak Response")
        self.tts_btn.clicked.connect(self.on_speak_last)
        self.record_btn = QPushButton("Record (STT)")
        self.record_btn.clicked.connect(self.on_record)
        btns = QHBoxLayout()
        btns.addWidget(self.send_btn)
        btns.addWidget(self.record_btn)
        btns.addWidget(self.tts_btn)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.chat_input)
        layout.addLayout(btns)
        w.setLayout(layout)
        return w

    def on_send(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        ts = time.time()
        self._append_message({"role":"user","text":text,"ts":ts})
        self.chat_input.clear()
        # generate response
        try:
            resp = self.engine.generate(text, max_gen=40)
        except Exception as e:
            resp = f"[Error generating: {e}]"
        self._append_message({"role":"assistant","text":resp,"ts":time.time()})
        # auto-save to memory DB
        from ..memory.memory_db import add_memory
        add_memory(text, source="user", meta={"via":"gui"})
        add_memory(resp, source="assistant", meta={"via":"gui"})
        # update logbook list
        self._refresh_logbook_list()

    def on_speak_last(self):
        # speak the last assistant message if present
        last = None
        for m in reversed(self.current_messages):
            if m.get("role") == "assistant":
                last = m.get("text")
                break
        if last:
            speak(last)

    def on_record(self):
        # run in thread to avoid blocking UI
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Recording...")
        self.stt_thread = STTThread(timeout=8)
        self.stt_thread.result.connect(self._on_record_result)
        self.stt_thread.start()

    def _on_record_result(self, txt):
        self.record_btn.setEnabled(True)
        self.record_btn.setText("Record (STT)")
        if not txt:
            QMessageBox.information(self, "STT", "No speech recognized or STT not available.")
            return
        self.chat_input.setText(txt)

    def _append_message(self, msg: dict):
        role = msg.get("role")
        text = msg.get("text")
        ts = msg.get("ts", time.time())
        nice = f"[{role} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] {text}"
        self.chat_display.append(nice)
        self.current_messages.append(msg)

    # Logbook Tab
    def _logbook_tab(self):
        w = QWidget()
        layout = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.conv_list = QListWidget()
        self.conv_list.itemDoubleClicked.connect(self.open_selected_conversation)
        left.addWidget(QLabel("Saved Conversations"))
        left.addWidget(self.conv_list)
        btn_load = QPushButton("Load")
        btn_save = QPushButton("Save Current As...")
        btn_refresh = QPushButton("Refresh")
        btn_delete = QPushButton("Delete")
        btn_load.clicked.connect(self.open_selected_conversation)
        btn_save.clicked.connect(self.save_current_conversation)
        btn_refresh.clicked.connect(self._refresh_logbook_list)
        btn_delete.clicked.connect(self.delete_selected_conversation)
        left.addWidget(btn_refresh)
        left.addWidget(btn_save)
        left.addWidget(btn_delete)
        right.addWidget(QLabel("Conversation Preview"))
        self.conv_preview = QTextEdit(readOnly=True)
        right.addWidget(self.conv_preview)
        layout.addLayout(left, 1)
        layout.addLayout(right, 2)
        w.setLayout(layout)
        self._refresh_logbook_list()
        return w

    def _refresh_logbook_list(self):
        names = self.conv_manager.list_conversations()
        self.conv_list.clear()
        for n in names:
            self.conv_list.addItem(n)

    def open_selected_conversation(self):
        item = self.conv_list.currentItem()
        if not item:
            return
        name = item.text()
        try:
            conv = self.conv_manager.load_conversation(name)
            preview = json.dumps(conv, indent=2)
            self.conv_preview.setText(preview)
            # Optionally load into chat view
            self.chat_display.clear()
            self.current_messages = conv.get("messages", [])
            for m in self.current_messages:
                self._append_message(m)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))

    def save_current_conversation(self):
        name, ok = QFileDialog.getSaveFileName(self, "Save Conversation As", str(CONFIG["data_dir"]))
        if not name:
            return
        # ensure .json extension
        if not name.endswith(".json"):
            name += ".json"
        # create name from filename
        base = Path(name).stem
        self.conv_manager.save_conversation(base, self.current_messages)
        self._refresh_logbook_list()
        QMessageBox.information(self, "Saved", f"Saved conversation as {base}.json")

    def delete_selected_conversation(self):
        item = self.conv_list.currentItem()
        if not item:
            return
        name = item.text()
        path = self.conv_manager.conv_dir / f"{name}.json"
        if path.exists():
            path.unlink()
        self._refresh_logbook_list()

    # Training tab (placeholder)
    def _training_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Training controls (placeholder)"))
        btn_train = QPushButton("Run Quick Train (toy)")
        btn_train.clicked.connect(self._run_quick_train)
        layout.addWidget(btn_train)
        w.setLayout(layout)
        return w

    def _run_quick_train(self):
        from ..core.training import train_model
        # run training in a new thread? For simplicity just call (blocks UI). Could be improved.
        train_model(force=False, num_epochs=2)
        QMessageBox.information(self, "Training", "Quick training finished (toy).")

    # Avatar tab
    def _avatar_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Avatar Controller"))
        self.avatar_x = QLineEdit("0")
        self.avatar_y = QLineEdit("0")
        btn_move = QPushButton("Move Avatar")
        btn_move.clicked.connect(self._move_avatar)
        btn_speak = QPushButton("Avatar Speak (TTS)")
        btn_speak.clicked.connect(self._avatar_speak)
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.avatar_x)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.avatar_y)
        layout.addWidget(btn_move)
        layout.addWidget(btn_speak)
        w.setLayout(layout)
        return w

    def _move_avatar(self):
        try:
            x = int(self.avatar_x.text())
            y = int(self.avatar_y.text())
            self.avatar.move(x, y)
            QMessageBox.information(self, "Avatar", f"Moved avatar to ({x},{y})")
        except Exception as e:
            QMessageBox.warning(self, "Avatar", f"Failed: {e}")

    def _avatar_speak(self):
        # speak last assistant message via avatar
        last = None
        for m in reversed(self.current_messages):
            if m.get("role") == "assistant":
                last = m.get("text")
                break
        if last:
            # call avatar.speak (which uses TTS)
            self.avatar.speak(last)
            QMessageBox.information(self, "Avatar", "Avatar spoke the last response.")

def run_app():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
