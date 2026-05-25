"""TC-8: Tests for download_progress.py — progress tracking, formatting."""
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enigma_engine.core.download_progress import (
    DownloadProgress,
    DownloadState,
    ProgressCallback,
    ProgressState,
    format_bytes,
)


# ================================================================
# format_bytes
# ================================================================


class TestFormatBytes:
    """Test human-readable byte formatting."""

    @pytest.mark.parametrize("size, expected", [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
        (1099511627776, "1.0 TB"),
    ])
    def test_format_bytes(self, size, expected):
        assert format_bytes(size) == expected


# ================================================================
# DownloadState enum
# ================================================================


class TestDownloadState:
    """Test DownloadState enum values exist."""

    def test_all_states_exist(self):
        assert DownloadState.PENDING
        assert DownloadState.DOWNLOADING
        assert DownloadState.COMPLETED
        assert DownloadState.FAILED
        assert DownloadState.CANCELLED

    def test_states_are_distinct(self):
        states = [DownloadState.PENDING, DownloadState.DOWNLOADING,
                  DownloadState.COMPLETED, DownloadState.FAILED,
                  DownloadState.CANCELLED]
        assert len(set(states)) == 5


# ================================================================
# ProgressState
# ================================================================


class TestProgressState:
    """Test ProgressState dataclass."""

    def test_defaults(self):
        ps = ProgressState()
        assert ps.task_name == ""
        assert ps.total is None
        assert ps.current == 0
        assert ps.status == ""
        assert ps.started_at is None
        assert ps.finished_at is None

    def test_custom_values(self):
        ps = ProgressState(task_name="download", total=100, current=50)
        assert ps.task_name == "download"
        assert ps.total == 100
        assert ps.current == 50


# ================================================================
# DownloadProgress
# ================================================================


class TestDownloadProgress:
    """Test DownloadProgress properties."""

    def test_percentage_zero(self):
        dp = DownloadProgress(file_size=0, downloaded=0)
        assert dp.percentage == 0.0

    def test_percentage_half(self):
        dp = DownloadProgress(file_size=1000, downloaded=500)
        assert dp.percentage == pytest.approx(50.0)

    def test_percentage_caps_at_100(self):
        dp = DownloadProgress(file_size=100, downloaded=200)
        assert dp.percentage == 100.0

    def test_speed_str(self):
        dp = DownloadProgress(speed=1048576)
        assert dp.speed_str == "1.0 MB/s"

    def test_eta_str_calculating(self):
        dp = DownloadProgress(eta=0)
        assert dp.eta_str == "calculating..."

    def test_eta_str_seconds(self):
        dp = DownloadProgress(eta=30)
        assert dp.eta_str == "30s"

    def test_eta_str_minutes(self):
        dp = DownloadProgress(eta=135)  # 2m 15s
        assert dp.eta_str == "2m 15s"

    def test_eta_str_hours(self):
        dp = DownloadProgress(eta=5400)  # 1h 30m
        assert dp.eta_str == "1h 30m"

    def test_size_str(self):
        dp = DownloadProgress(file_size=1048576, downloaded=524288)
        assert dp.size_str == "512.0 KB / 1.0 MB"

    def test_to_progress_state(self):
        dp = DownloadProgress(
            file_name="model.bin", file_size=1000, downloaded=500,
            state=DownloadState.DOWNLOADING, speed=100,
        )
        ps = dp.to_progress_state()
        assert isinstance(ps, ProgressState)
        assert "model.bin" in ps.task_name
        assert ps.total == 1000
        assert ps.current == 500

    def test_to_progress_state_completed(self):
        dp = DownloadProgress(state=DownloadState.COMPLETED)
        ps = dp.to_progress_state()
        assert ps.finished_at is not None

    def test_defaults(self):
        dp = DownloadProgress()
        assert dp.file_name == ""
        assert dp.state == DownloadState.PENDING
        assert dp.error is None
        assert dp.current_file == 0
        assert dp.total_files == 0


# ================================================================
# ProgressCallback
# ================================================================


class TestProgressCallback:
    """Test ProgressCallback behavior."""

    def test_callback_fires(self):
        received = []
        cb = ProgressCallback(callback=lambda p: received.append(p))
        cb.set_file_name("test.bin")
        # Simulate a download: first call initializes
        cb(100, 1000, 100)
        assert len(received) >= 1
        assert received[0].file_name == "test.bin"

    def test_state_transitions(self):
        states = []
        cb = ProgressCallback(
            callback=lambda p: states.append(p.state),
            show_cli=False,
        )
        cb(50, 100, 50)    # first call → DOWNLOADING
        assert DownloadState.DOWNLOADING in states

    def test_state_completes(self):
        """State becomes COMPLETED when downloaded >= total."""
        cb = ProgressCallback(show_cli=False)
        cb(50, 100, 50)
        cb(50, 100, 100)
        assert cb._progress.state == DownloadState.COMPLETED

    def test_set_file_count(self):
        cb = ProgressCallback(show_cli=False)
        cb.set_file_count(2, 5)
        assert cb._progress.current_file == 2
        assert cb._progress.total_files == 5

    def test_reset(self):
        cb = ProgressCallback(show_cli=False)
        cb.set_file_name("a.bin")
        cb(100, 1000, 100)
        cb.reset()
        # After reset, internal state is cleared
        assert cb._start_time == 0.0

    def test_no_callback_no_crash(self):
        """Callback=None should not raise."""
        cb = ProgressCallback(callback=None, show_cli=False)
        cb(100, 1000, 100)

    def test_speed_calculation(self):
        """Speed should be calculated after enough time passes."""
        speeds = []
        cb = ProgressCallback(
            callback=lambda p: speeds.append(p.speed),
            show_cli=False,
        )
        cb(0, 10000, 0)
        # Simulate time passing for speed calc
        cb._last_update = time.time() - 1.0
        cb._last_downloaded = 0
        cb(5000, 10000, 5000)
        # At least one speed entry should be > 0
        assert any(s > 0 for s in speeds)


# ================================================================
# DownloadTracker.download_model — global progress-bar state
# ================================================================


class TestDownloadModelProgressBarRestore:
    """Pass 156z9en regression: ``disable_progress_bars()`` is global
    state. If ``snapshot_download`` raises and we never call the
    paired ``enable_progress_bars()``, HF progress bars stay disabled
    for the rest of the process — affecting unrelated later downloads.
    The pairing must hold on every exit path, not only the success
    path.
    """

    def test_progress_bars_re_enabled_when_download_raises(
            self, monkeypatch):
        from enigma_engine.core.download_progress import DownloadTracker

        calls: list[str] = []

        fake_hub = type(sys)("huggingface_hub")
        fake_hub_utils = type(sys)("huggingface_hub.utils")

        def _fake_disable():
            calls.append("disable")

        def _fake_enable():
            calls.append("enable")

        def _fake_snapshot_download(**_kwargs):
            calls.append("download")
            raise RuntimeError("simulated network failure")

        fake_hub.snapshot_download = _fake_snapshot_download
        fake_hub_utils.disable_progress_bars = _fake_disable
        fake_hub_utils.enable_progress_bars = _fake_enable
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        monkeypatch.setitem(
            sys.modules, "huggingface_hub.utils", fake_hub_utils)

        tracker = DownloadTracker(callback=None, show_cli=False)
        result = tracker.download_model("fake/model")

        assert result is None
        # Disable must be paired with enable on the failure path.
        assert "disable" in calls
        assert "enable" in calls, (
            "enable_progress_bars() never called after failure — "
            "global state leak")
        # Enable must happen after the failed download, not before it.
        assert calls.index("enable") > calls.index("download")
