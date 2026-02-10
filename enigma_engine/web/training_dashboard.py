"""
Web Training Dashboard for Enigma AI Engine

Browser-based training management interface.

Features:
- Training job monitoring
- Dataset management
- Model configuration
- Live metrics visualization
- Checkpoint downloads

Usage:
    from enigma_engine.web.training_dashboard import TrainingDashboard
    
    dashboard = TrainingDashboard()
    dashboard.run(port=5001)  # Access at http://localhost:5001
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics snapshot."""
    timestamp: float
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    memory_used: float = 0.0


@dataclass
class TrainingSession:
    """A training session."""
    session_id: str
    model_name: str
    dataset_name: str
    status: str = "idle"  # idle, running, paused, completed, error
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    
    # Progress
    current_epoch: int = 0
    total_epochs: int = 10
    current_step: int = 0
    total_steps: int = 0
    
    # Metrics history
    metrics: List[TrainingMetrics] = field(default_factory=list)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)


class TrainingDashboard:
    """Web-based training dashboard."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize dashboard.
        
        Args:
            data_dir: Directory for datasets and checkpoints
        """
        self.data_dir = data_dir or Path("data")
        self._sessions: Dict[str, TrainingSession] = {}
        self._current_session: Optional[str] = None
        self._app = None
    
    def _create_app(self):
        """Create Flask application."""
        try:
            from flask import Flask, jsonify, request, render_template_string
        except ImportError:
            logger.error("Flask not installed. Run: pip install flask")
            raise ImportError("Flask required for dashboard")
        
        app = Flask(__name__)
        
        # Main dashboard page
        @app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        # API endpoints
        @app.route('/api/sessions', methods=['GET'])
        def list_sessions():
            sessions = []
            for session in self._sessions.values():
                sessions.append({
                    "session_id": session.session_id,
                    "model_name": session.model_name,
                    "dataset_name": session.dataset_name,
                    "status": session.status,
                    "progress": (session.current_step / max(1, session.total_steps)) * 100,
                    "current_epoch": session.current_epoch,
                    "total_epochs": session.total_epochs
                })
            return jsonify(sessions)
        
        @app.route('/api/sessions/<session_id>', methods=['GET'])
        def get_session(session_id):
            session = self._sessions.get(session_id)
            if not session:
                return jsonify({"error": "Session not found"}), 404
            
            return jsonify({
                "session_id": session.session_id,
                "model_name": session.model_name,
                "dataset_name": session.dataset_name,
                "status": session.status,
                "current_epoch": session.current_epoch,
                "total_epochs": session.total_epochs,
                "current_step": session.current_step,
                "total_steps": session.total_steps,
                "config": session.config,
                "created_at": session.created_at
            })
        
        @app.route('/api/sessions/<session_id>/metrics', methods=['GET'])
        def get_metrics(session_id):
            session = self._sessions.get(session_id)
            if not session:
                return jsonify({"error": "Session not found"}), 404
            
            # Return recent metrics
            metrics = []
            for m in session.metrics[-100:]:
                metrics.append({
                    "timestamp": m.timestamp,
                    "step": m.step,
                    "loss": m.loss,
                    "learning_rate": m.learning_rate
                })
            
            return jsonify(metrics)
        
        @app.route('/api/sessions', methods=['POST'])
        def create_session():
            data = request.json or {}
            
            session = TrainingSession(
                session_id=str(uuid.uuid4())[:8],
                model_name=data.get("model_name", "forge-small"),
                dataset_name=data.get("dataset_name", "default"),
                config=data.get("config", {})
            )
            
            self._sessions[session.session_id] = session
            
            return jsonify({"session_id": session.session_id})
        
        @app.route('/api/sessions/<session_id>/start', methods=['POST'])
        def start_training(session_id):
            session = self._sessions.get(session_id)
            if not session:
                return jsonify({"error": "Session not found"}), 404
            
            session.status = "running"
            session.started_at = time.time()
            self._current_session = session_id
            
            return jsonify({"status": "started"})
        
        @app.route('/api/sessions/<session_id>/stop', methods=['POST'])
        def stop_training(session_id):
            session = self._sessions.get(session_id)
            if not session:
                return jsonify({"error": "Session not found"}), 404
            
            session.status = "paused"
            
            return jsonify({"status": "stopped"})
        
        @app.route('/api/datasets', methods=['GET'])
        def list_datasets():
            datasets = []
            data_dir = self.data_dir / "training"
            
            if data_dir.exists():
                for f in data_dir.glob("*.txt"):
                    datasets.append({
                        "name": f.stem,
                        "path": str(f),
                        "size_mb": f.stat().st_size / (1024 * 1024)
                    })
            
            return jsonify(datasets)
        
        @app.route('/api/models', methods=['GET'])
        def list_models():
            models = [
                {"name": "forge-nano", "params": "1M"},
                {"name": "forge-micro", "params": "2M"},
                {"name": "forge-tiny", "params": "5M"},
                {"name": "forge-small", "params": "27M"},
                {"name": "forge-medium", "params": "85M"},
                {"name": "forge-large", "params": "300M"}
            ]
            return jsonify(models)
        
        return app
    
    def add_metrics(self, session_id: str, metrics: TrainingMetrics):
        """Add training metrics to a session."""
        session = self._sessions.get(session_id)
        if session:
            session.metrics.append(metrics)
            session.current_step = metrics.step
            session.current_epoch = metrics.epoch
    
    def update_progress(
        self,
        session_id: str,
        step: int,
        loss: float,
        epoch: int = 0
    ):
        """Update training progress."""
        session = self._sessions.get(session_id)
        if session:
            session.current_step = step
            session.current_epoch = epoch
            
            # Add metric
            metrics = TrainingMetrics(
                timestamp=time.time(),
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=session.config.get("learning_rate", 0.0001)
            )
            session.metrics.append(metrics)
    
    def run(self, host: str = "0.0.0.0", port: int = 5001, debug: bool = False):
        """
        Run the dashboard server.
        
        Args:
            host: Host to bind to
            port: Port number
            debug: Enable debug mode
        """
        self._app = self._create_app()
        logger.info(f"Starting training dashboard at http://{host}:{port}")
        self._app.run(host=host, port=port, debug=debug)


# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Enigma Training Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a2e; color: #eee; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        .card { background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .card h2 { color: #00d4ff; margin-bottom: 15px; font-size: 18px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .stat { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #2a2a4e; }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #888; }
        .stat-value { font-weight: bold; }
        .btn { background: #00d4ff; color: #000; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #00b8e0; }
        .btn-danger { background: #ff4444; color: #fff; }
        .progress-bar { background: #2a2a4e; height: 20px; border-radius: 10px; overflow: hidden; }
        .progress-fill { background: linear-gradient(90deg, #00d4ff, #00ff88); height: 100%; transition: width 0.3s; }
        select, input { background: #2a2a4e; border: 1px solid #3a3a5e; color: #eee; padding: 8px; border-radius: 4px; width: 100%; margin-bottom: 10px; }
        .sessions { margin-top: 20px; }
        .session-item { background: #1f2847; padding: 15px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .session-info { flex: 1; }
        .session-name { font-weight: bold; }
        .session-status { padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .status-running { background: #00ff88; color: #000; }
        .status-idle { background: #666; }
        .status-completed { background: #00d4ff; color: #000; }
        .chart { height: 200px; background: #1f2847; border-radius: 8px; display: flex; align-items: flex-end; padding: 10px; gap: 2px; }
        .bar { background: #00d4ff; width: 10px; min-height: 5px; border-radius: 2px 2px 0 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enigma Training Dashboard</h1>
        
        <div class="grid">
            <div class="card">
                <h2>New Training Session</h2>
                <select id="model-select">
                    <option value="forge-small">Forge Small (27M)</option>
                    <option value="forge-medium">Forge Medium (85M)</option>
                    <option value="forge-large">Forge Large (300M)</option>
                </select>
                <select id="dataset-select">
                    <option value="">Select Dataset...</option>
                </select>
                <input type="number" id="epochs" placeholder="Epochs" value="10">
                <input type="number" id="batch-size" placeholder="Batch Size" value="32">
                <button class="btn" onclick="createSession()">Create Session</button>
            </div>
            
            <div class="card">
                <h2>Current Training</h2>
                <div id="current-training">
                    <p style="color: #888;">No active training</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Loss Chart</h2>
            <div class="chart" id="loss-chart">
                <!-- Bars will be added dynamically -->
            </div>
        </div>
        
        <div class="card">
            <h2>Training Sessions</h2>
            <div class="sessions" id="sessions-list">
                <p style="color: #888;">No sessions yet</p>
            </div>
        </div>
    </div>
    
    <script>
        async function loadSessions() {
            const resp = await fetch('/api/sessions');
            const sessions = await resp.json();
            
            const list = document.getElementById('sessions-list');
            if (sessions.length === 0) {
                list.innerHTML = '<p style="color: #888;">No sessions yet</p>';
                return;
            }
            
            list.innerHTML = sessions.map(s => `
                <div class="session-item">
                    <div class="session-info">
                        <div class="session-name">${s.model_name} - ${s.dataset_name}</div>
                        <div style="color: #888; font-size: 12px;">
                            Epoch ${s.current_epoch}/${s.total_epochs} | 
                            Progress: ${s.progress.toFixed(1)}%
                        </div>
                    </div>
                    <span class="session-status status-${s.status}">${s.status}</span>
                    <button class="btn" onclick="startSession('${s.session_id}')" style="margin-left: 10px;">
                        ${s.status === 'running' ? 'Stop' : 'Start'}
                    </button>
                </div>
            `).join('');
        }
        
        async function loadDatasets() {
            const resp = await fetch('/api/datasets');
            const datasets = await resp.json();
            
            const select = document.getElementById('dataset-select');
            select.innerHTML = '<option value="">Select Dataset...</option>' +
                datasets.map(d => `<option value="${d.name}">${d.name} (${d.size_mb.toFixed(1)} MB)</option>`).join('');
        }
        
        async function createSession() {
            const model = document.getElementById('model-select').value;
            const dataset = document.getElementById('dataset-select').value;
            const epochs = document.getElementById('epochs').value;
            const batchSize = document.getElementById('batch-size').value;
            
            if (!dataset) {
                alert('Please select a dataset');
                return;
            }
            
            const resp = await fetch('/api/sessions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    model_name: model,
                    dataset_name: dataset,
                    config: {epochs: parseInt(epochs), batch_size: parseInt(batchSize)}
                })
            });
            
            const result = await resp.json();
            alert('Session created: ' + result.session_id);
            loadSessions();
        }
        
        async function startSession(sessionId) {
            await fetch(`/api/sessions/${sessionId}/start`, {method: 'POST'});
            loadSessions();
        }
        
        // Initial load
        loadSessions();
        loadDatasets();
        
        // Auto-refresh
        setInterval(loadSessions, 5000);
    </script>
</body>
</html>
"""


# Convenience function
def run_training_dashboard(port: int = 5001):
    """Run the training dashboard."""
    dashboard = TrainingDashboard()
    dashboard.run(port=port)
