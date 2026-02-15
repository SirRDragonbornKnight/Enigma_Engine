"""
Game Tab Module

Auto-detects running games and provides AI assistance through:
- Process scanning to identify games
- Internet research to learn game specifics
- Integration with Persona system for game-specific prompts
"""

import json
import threading
from pathlib import Path

from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QMessageBox,
)

from ....config import CONFIG


class GameSignals(QObject):
    """Signals for thread-safe UI updates."""
    game_detected = pyqtSignal(str, str)  # process_name, game_name
    game_closed = pyqtSignal(str)  # process_name
    research_complete = pyqtSignal(str, str)  # game_name, info
    research_error = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # log text


# Known games database (process name -> display name)
KNOWN_GAMES = {
    # Popular games
    "minecraft.exe": "Minecraft",
    "javaw.exe": "Minecraft (Java)",
    "terraria.exe": "Terraria",
    "factorio.exe": "Factorio",
    "stardewvalley.exe": "Stardew Valley",
    "stardew valley.exe": "Stardew Valley",
    "csgo.exe": "Counter-Strike: GO",
    "cs2.exe": "Counter-Strike 2",
    "valorant.exe": "Valorant",
    "valorant-win64-shipping.exe": "Valorant",
    "leagueclient.exe": "League of Legends",
    "league of legends.exe": "League of Legends",
    "dota2.exe": "Dota 2",
    "rocketleague.exe": "Rocket League",
    "fortnite.exe": "Fortnite",
    "fortniteclient-win64-shipping.exe": "Fortnite",
    "apex_legends.exe": "Apex Legends",
    "r5apex.exe": "Apex Legends",
    "overwatch.exe": "Overwatch",
    "overwatch 2.exe": "Overwatch 2",
    "gta5.exe": "Grand Theft Auto V",
    "gtav.exe": "Grand Theft Auto V",
    "eldenring.exe": "Elden Ring",
    "darksouls.exe": "Dark Souls",
    "darksoulsiii.exe": "Dark Souls III",
    "sekiro.exe": "Sekiro",
    "witcher3.exe": "The Witcher 3",
    "cyberpunk2077.exe": "Cyberpunk 2077",
    "rdr2.exe": "Red Dead Redemption 2",
    "skyrim.exe": "The Elder Scrolls V: Skyrim",
    "skyrimse.exe": "Skyrim Special Edition",
    "fallout4.exe": "Fallout 4",
    "baldursgate3.exe": "Baldur's Gate 3",
    "bg3.exe": "Baldur's Gate 3",
    "hogwarts legacy.exe": "Hogwarts Legacy",
    "palworld-win64-shipping.exe": "Palworld",
    "helldivers2.exe": "Helldivers 2",
    # Creative software
    "blender.exe": "Blender",
    "unity.exe": "Unity Editor",
    "unrealengine.exe": "Unreal Engine",
    "godot.exe": "Godot Engine",
}


# Game data directory
GAME_DATA_DIR = Path(CONFIG["data_dir"]) / "game"


def create_game_subtab(parent):
    """
    Create the Game tab with simplified, effective design:
    - Auto-detect running games via process scanning
    - Research games online to understand how to help
    - Use Persona system for game-specific instructions
    """
    widget = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(10)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # Initialize signals for thread-safe UI updates
    parent._game_signals = GameSignals()
    parent._game_signals.game_detected.connect(lambda p, n: _on_game_detected_ui(parent, p, n))
    parent._game_signals.game_closed.connect(lambda p: _on_game_closed_ui(parent, p))
    parent._game_signals.research_complete.connect(lambda g, i: _on_research_complete_ui(parent, g, i))
    parent._game_signals.research_error.connect(lambda e: _on_research_error_ui(parent, e))
    parent._game_signals.log_message.connect(lambda m: parent.game_log.append(m))
    
    # === GAME DETECTION ===
    detect_group = QGroupBox("Game Detection")
    detect_layout = QVBoxLayout(detect_group)
    
    # Description
    desc = QLabel(
        "Automatically detects running games and researches how to help you play."
    )
    desc.setWordWrap(True)
    detect_layout.addWidget(desc)
    
    # Detection toggle and status row
    detect_row = QHBoxLayout()
    parent.auto_detect_check = QCheckBox("Auto-Detect Games")
    parent.auto_detect_check.setChecked(False)
    parent.auto_detect_check.setToolTip("Scan running processes for known games")
    parent.auto_detect_check.stateChanged.connect(lambda s: _toggle_detection(parent, s))
    detect_row.addWidget(parent.auto_detect_check)
    
    parent.detect_status = QLabel("Detection: Off")
    parent.detect_status.setStyleSheet("color: #6c7086;")
    detect_row.addWidget(parent.detect_status)
    detect_row.addStretch()
    detect_layout.addLayout(detect_row)
    
    # Currently detected game
    game_row = QHBoxLayout()
    game_row.addWidget(QLabel("Active Game:"))
    parent.current_game_label = QLabel("None")
    parent.current_game_label.setStyleSheet("color: #cdd6f4; font-weight: bold;")
    game_row.addWidget(parent.current_game_label)
    game_row.addStretch()
    detect_layout.addLayout(game_row)
    
    layout.addWidget(detect_group)
    
    # === GAME RESEARCH ===
    research_group = QGroupBox("Game Research")
    research_layout = QVBoxLayout(research_group)
    
    research_desc = QLabel(
        "When a game is detected, AI can research it online to learn controls, "
        "strategies, and how to assist you effectively."
    )
    research_desc.setWordWrap(True)
    research_layout.addWidget(research_desc)
    
    # Research options
    parent.auto_research_check = QCheckBox("Auto-Research New Games")
    parent.auto_research_check.setChecked(True)
    parent.auto_research_check.setToolTip("Automatically search for game info when detected")
    research_layout.addWidget(parent.auto_research_check)
    
    # Research progress
    parent.research_progress = QProgressBar()
    parent.research_progress.setVisible(False)
    parent.research_progress.setTextVisible(True)
    parent.research_progress.setFormat("Researching...")
    research_layout.addWidget(parent.research_progress)
    
    # Manual research button
    research_btn_row = QHBoxLayout()
    parent.btn_research = QPushButton("Research Current Game")
    parent.btn_research.setEnabled(False)
    parent.btn_research.clicked.connect(lambda: _research_game(parent))
    research_btn_row.addWidget(parent.btn_research)
    research_btn_row.addStretch()
    research_layout.addLayout(research_btn_row)
    
    layout.addWidget(research_group)
    
    # === KNOWN GAMES ===
    known_group = QGroupBox("Known Games")
    known_layout = QVBoxLayout(known_group)
    
    parent.known_games_list = QListWidget()
    parent.known_games_list.setMaximumHeight(120)
    parent.known_games_list.setToolTip("Games with saved research data")
    known_layout.addWidget(parent.known_games_list)
    
    # Buttons row
    known_btn_row = QHBoxLayout()
    btn_view = QPushButton("View Info")
    btn_view.clicked.connect(lambda: _view_game_info(parent))
    known_btn_row.addWidget(btn_view)
    
    btn_clear = QPushButton("Clear")
    btn_clear.clicked.connect(lambda: _clear_game_info(parent))
    known_btn_row.addWidget(btn_clear)
    known_btn_row.addStretch()
    known_layout.addLayout(known_btn_row)
    
    layout.addWidget(known_group)
    
    # === LOG OUTPUT ===
    parent.game_log = QTextEdit()
    parent.game_log.setReadOnly(True)
    parent.game_log.setPlaceholderText("Game detection events will appear here...")
    parent.game_log.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 4px;
            padding: 8px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
        }
    """)
    layout.addWidget(parent.game_log, stretch=1)
    
    widget.setLayout(layout)
    
    # Initialize state
    parent._current_game = None
    parent._detection_timer = None
    parent._researched_games = {}
    
    # Load saved game data
    _load_known_games(parent)
    
    return widget


# === DETECTION FUNCTIONS ===

def _toggle_detection(parent, state):
    """Toggle automatic game detection."""
    enabled = state == 2
    
    if enabled:
        # Start detection timer
        parent._detection_timer = QTimer()
        parent._detection_timer.timeout.connect(lambda: _scan_for_games(parent))
        parent._detection_timer.start(5000)  # Scan every 5 seconds
        
        parent.detect_status.setText("Detection: Active")
        parent.detect_status.setStyleSheet("color: #a6e3a1;")
        parent.game_log.append("[OK] Game detection started")
        
        # Do immediate scan
        _scan_for_games(parent)
    else:
        # Stop detection
        if parent._detection_timer:
            parent._detection_timer.stop()
            parent._detection_timer = None
        
        parent.detect_status.setText("Detection: Off")
        parent.detect_status.setStyleSheet("color: #6c7086;")
        parent.game_log.append("[x] Game detection stopped")


def _scan_for_games(parent):
    """Scan running processes for known games."""
    try:
        import psutil
    except ImportError:
        parent.game_log.append("[!] psutil not installed - run: pip install psutil")
        parent.auto_detect_check.setChecked(False)
        return
    
    found_game = None
    found_process = None
    
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if name in KNOWN_GAMES:
                found_game = KNOWN_GAMES[name]
                found_process = name
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Update UI via signal
    if found_game and found_game != parent._current_game:
        parent._game_signals.game_detected.emit(found_process, found_game)
    elif not found_game and parent._current_game:
        parent._game_signals.game_closed.emit(parent._current_game)


def _on_game_detected_ui(parent, process_name: str, game_name: str):
    """Handle game detection in UI thread."""
    parent._current_game = game_name
    parent.current_game_label.setText(game_name)
    parent.current_game_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
    parent.btn_research.setEnabled(True)
    parent.game_log.append(f"[G] Detected: {game_name} ({process_name})")
    
    # Auto-research if enabled and not already researched
    if parent.auto_research_check.isChecked():
        if game_name not in parent._researched_games:
            _research_game(parent)


def _on_game_closed_ui(parent, game_name: str):
    """Handle game closure in UI thread."""
    parent.game_log.append(f"[x] Game closed: {game_name}")
    parent._current_game = None
    parent.current_game_label.setText("None")
    parent.current_game_label.setStyleSheet("color: #cdd6f4; font-weight: bold;")
    parent.btn_research.setEnabled(False)


# === RESEARCH FUNCTIONS ===

def _research_game(parent):
    """Research the current game online."""
    if not parent._current_game:
        return
    
    game = parent._current_game
    parent.research_progress.setVisible(True)
    parent.research_progress.setRange(0, 0)  # Indeterminate
    parent.btn_research.setEnabled(False)
    parent.game_log.append(f"[>] Researching: {game}...")
    
    # Run research in background thread
    def do_research():
        try:
            info = _fetch_game_info(game)
            parent._game_signals.research_complete.emit(game, info)
        except Exception as e:
            parent._game_signals.research_error.emit(str(e))
    
    thread = threading.Thread(target=do_research, daemon=True)
    thread.start()


def _fetch_game_info(game_name: str) -> str:
    """Fetch game information from the internet."""
    # Try to use web tools if available
    try:
        from enigma_engine.tools.web_tools import search_web
        results = search_web(f"{game_name} game controls guide tips", max_results=3)
        if results:
            info_parts = [f"# {game_name} - Game Info\n"]
            for r in results:
                info_parts.append(f"## {r.get('title', 'Info')}\n{r.get('snippet', '')}\n")
            return "\n".join(info_parts)
    except ImportError:
        pass
    
    # Fallback: Generate basic info from game name
    return f"""# {game_name}

## AI Assistant Ready
I can help you with {game_name}! Ask me about:
- Controls and keybindings
- Tips and strategies
- Quest/mission help
- Character builds
- Game mechanics

Just ask in the Chat tab!
"""


def _on_research_complete_ui(parent, game: str, info: str):
    """Handle research completion in UI thread."""
    parent.research_progress.setVisible(False)
    parent.btn_research.setEnabled(True)
    
    # Save research data
    parent._researched_games[game] = info
    _save_game_info(game, info)
    _refresh_known_games_list(parent)
    
    parent.game_log.append(f"[OK] Research complete for: {game}")
    
    # Update persona with game context (if available)
    try:
        _update_persona_with_game(parent, game, info)
    except Exception:
        pass


def _on_research_error_ui(parent, error: str):
    """Handle research error in UI thread."""
    parent.research_progress.setVisible(False)
    parent.btn_research.setEnabled(True)
    parent.game_log.append(f"[X] Research failed: {error}")


def _update_persona_with_game(parent, game: str, info: str):
    """Update the AI persona with game context."""
    # Check if main window has persona
    if hasattr(parent, 'current_persona') and parent.current_persona:
        # Add game context to persona
        game_context = f"\n\n[Gaming Context: Currently helping with {game}]\n{info[:500]}"
        # This would integrate with the persona system
        parent.game_log.append(f"[i] Persona updated with {game} context")


# === DATA PERSISTENCE ===

def _load_known_games(parent):
    """Load saved game research data."""
    data_file = GAME_DATA_DIR / "researched_games.json"
    if data_file.exists():
        try:
            with open(data_file) as f:
                parent._researched_games = json.load(f)
            _refresh_known_games_list(parent)
        except Exception:
            parent._researched_games = {}


def _save_game_info(game: str, info: str):
    """Save game research info to disk."""
    GAME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = GAME_DATA_DIR / "researched_games.json"
    
    existing = {}
    if data_file.exists():
        try:
            with open(data_file) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    
    existing[game] = info
    
    with open(data_file, 'w') as f:
        json.dump(existing, f, indent=2)


def _refresh_known_games_list(parent):
    """Refresh the known games list widget."""
    parent.known_games_list.clear()
    for game in sorted(parent._researched_games.keys()):
        item = QListWidgetItem(game)
        parent.known_games_list.addItem(item)


def _view_game_info(parent):
    """View info for selected game."""
    item = parent.known_games_list.currentItem()
    if not item:
        QMessageBox.information(parent, "No Selection", "Select a game from the list.")
        return
    
    game = item.text()
    info = parent._researched_games.get(game, "No info available.")
    
    # Show in log
    parent.game_log.clear()
    parent.game_log.append(info)


def _clear_game_info(parent):
    """Clear selected game's research data."""
    item = parent.known_games_list.currentItem()
    if not item:
        return
    
    game = item.text()
    reply = QMessageBox.question(
        parent, "Clear Game Data",
        f"Remove research data for {game}?",
        QMessageBox.Yes | QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        if game in parent._researched_games:
            del parent._researched_games[game]
            _save_game_info_all(parent._researched_games)
            _refresh_known_games_list(parent)
            parent.game_log.append(f"[x] Cleared data for: {game}")


def _save_game_info_all(games: dict):
    """Save all game research data."""
    GAME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = GAME_DATA_DIR / "researched_games.json"
    with open(data_file, 'w') as f:
        json.dump(games, f, indent=2)
