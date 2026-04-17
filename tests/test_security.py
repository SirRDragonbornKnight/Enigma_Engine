"""Tests for safe model loading, SSRF protection, data validation, and thread safety."""
import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestSafeLoadWeightsSecurity:
    """Tests for safe model loading."""

    def test_safe_load_weights_missing_file(self):
        """safe_load_weights raises FileNotFoundError for missing files."""
        from enigma_engine.core.model_registry import safe_load_weights
        with pytest.raises(FileNotFoundError):
            safe_load_weights("/nonexistent/model.pth")

    def test_no_direct_torch_load_weights_only_false(self):
        """No code outside safe_load_weights should use torch.load(weights_only=False)."""
        import re
        root = Path(__file__).parent.parent / "enigma_engine"
        violations = []
        pattern = re.compile(r"torch\.load\(.*weights_only\s*=\s*False", re.DOTALL)
        for py in root.rglob("*.py"):
            source = py.read_text(encoding="utf-8")
            if pattern.search(source):
                violations.append(str(py.relative_to(root)))
        assert not violations, f"torch.load(weights_only=False) found in: {violations}"


class TestThreadSafety:
    """Verify thread safety locks and copy-on-write across modules."""

    # -- training_monitor.py --

    def test_training_monitor_has_lock(self):
        """TrainingMonitor must have a threading.Lock."""
        import threading
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        assert hasattr(m, "_lock")
        assert isinstance(m._lock, type(threading.Lock()))

    def test_training_monitor_losses_snapshot(self):
        """losses property must return a copy, not internal list."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        m.record_loss(1.0)
        snap = m.losses
        snap.append(999.0)
        assert 999.0 not in m.losses

    def test_training_monitor_chart_data_snapshot(self):
        """get_chart_data must return copies under the lock."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        m.record_loss(2.0)
        data = m.get_chart_data()
        data["losses"].append(999.0)
        assert 999.0 not in m.get_chart_data()["losses"]

    # -- model_registry.py --

    def test_model_registry_has_lock(self):
        """ModelRegistry must have a threading.Lock."""
        import threading
        from enigma_engine.core.model_registry import ModelRegistry
        r = ModelRegistry(models_dir="/tmp/nonexistent_model_dir")
        assert hasattr(r, "_lock")
        assert isinstance(r._lock, type(threading.Lock()))

    def test_model_registry_list_returns_copy(self):
        """list_models must return a copy of the registry dict."""
        from enigma_engine.core.model_registry import ModelRegistry
        r = ModelRegistry(models_dir="/tmp/nonexistent_model_dir")
        models = r.list_models()
        models["injected"] = {"bad": True}
        assert "injected" not in r.list_models()

    # -- hardware_detection.py --

    def test_hardware_detection_has_lock(self):
        """hardware_detection module must have _profile_lock."""
        import threading
        from enigma_engine.core import hardware_detection
        assert hasattr(hardware_detection, "_profile_lock")
        assert isinstance(
            hardware_detection._profile_lock, type(threading.Lock()))

    def test_detect_hardware_returns_consistent_copy(self):
        """detect_hardware() results must not be mutable singletons."""
        from enigma_engine.core.hardware_detection import (
            detect_hardware, clear_cached_profile)
        clear_cached_profile()
        p1 = detect_hardware()
        p2 = detect_hardware()
        # Both should be equal but modifying one shouldn't affect the other
        # (dataclass is immutable by field, but test the cache path)
        assert p1.device == p2.device
        assert p1.ram_gb == p2.ram_gb


# ================================================================
# GGUF FALLBACK REMOVAL — Suggestion #9A
# ================================================================

class TestDataValidation:
    """Test validate_training_data() function."""

    def test_valid_data(self):
        """Normal training text passes validation."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data(
            "Hello world this is a test.\nAnother line of training data.")
        assert result.is_valid is True
        assert result.total_sequences > 0
        assert len(result.errors) == 0

    def test_empty_data(self):
        """Empty string produces an error."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data("")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_short_sequences_warning(self):
        """Very short lines generate warnings."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data("a\nb\nc\nd\ne\nf\n")
        # short sequences should produce warnings
        assert len(result.warnings) > 0

    def test_duplicate_detection(self):
        """Duplicate lines are counted in stats."""
        from enigma_engine.core.training import validate_training_data

        text = "same line\n" * 10
        result = validate_training_data(text)
        assert result.stats.get("duplicates", 0) > 0

    def test_stats_populated(self):
        """Stats dict contains expected keys."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data(
            "Line one is long enough.\nLine two is also long enough.")
        assert "total_chars" in result.stats
        assert "total_lines" in result.stats
        assert "unique_lines" in result.stats
        assert "avg_length" in result.stats

    def test_null_bytes_warning(self):
        """Data with null bytes produces a warning."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data("Hello\x00World this is enough text")
        has_null_warning = any(
            "null" in w.lower() for w in result.warnings)
        assert has_null_warning

    def test_result_dataclass_fields(self):
        """DataValidationResult has the expected fields."""
        from enigma_engine.core.training import DataValidationResult

        r = DataValidationResult(
            is_valid=True,
            total_sequences=5,
            warnings=["warn"],
            errors=[],
            stats={"total_chars": 100},
        )
        assert r.is_valid is True
        assert r.total_sequences == 5
        assert len(r.warnings) == 1
        assert len(r.errors) == 0
        assert r.stats["total_chars"] == 100


class TestWebSSRF:
    """Test URL validation and response streaming in web_utils."""

    def test_validate_url_rejects_file_scheme(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_url("file:///etc/passwd")

    def test_validate_url_rejects_ftp_scheme(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_url("ftp://evil.com/payload")

    def test_validate_url_rejects_localhost(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="private|reserved"):
            _validate_url("http://127.0.0.1/admin")

    def test_validate_url_rejects_private_ip(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="private|reserved"):
            _validate_url("http://192.168.1.1/secret")

    def test_validate_url_rejects_no_hostname(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="No hostname"):
            _validate_url("http:///path")

    def test_max_response_bytes_constant(self):
        from enigma_engine.core.web_utils import _MAX_RESPONSE_BYTES

        assert _MAX_RESPONSE_BYTES == 1_048_576

    def test_fetch_page_text_validates_url(self):
        """fetch_page_text should reject private IPs before making a request."""
        from enigma_engine.core.web_utils import fetch_page_text

        with pytest.raises(ValueError, match="private|reserved"):
            fetch_page_text("http://10.0.0.1/internal")


# ================================================================
# Suggestion 20: _init_common eliminates attribute drift
# ================================================================


# ================================================================
# Consolidated Generation Paths
# ================================================================


# ================================================================
# CF-12: Config type validation on user config JSON
# ================================================================

class TestConfigTypeValidation:
    """User config JSON must be type-validated before merging."""

    def test_validate_config_types_exists(self):
        """_validate_config_types returns a dict and handles empty input."""
        from enigma_engine.config.defaults import _validate_config_types
        assert callable(_validate_config_types)
        result = _validate_config_types({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_rejects_string_port(self):
        """String value for api_port must be rejected."""
        from enigma_engine.config.defaults import _validate_config_types
        bad = {"api_port": "abc"}
        cleaned = _validate_config_types(bad)
        assert "api_port" not in cleaned

    def test_rejects_string_for_int_field(self):
        """String value for an int config field must be rejected."""
        from enigma_engine.config.defaults import _validate_config_types
        bad = {"batch_size": "not_a_number"}
        cleaned = _validate_config_types(bad)
        assert "batch_size" not in cleaned

    def test_accepts_valid_types(self):
        """Valid values should pass through unchanged."""
        from enigma_engine.config.defaults import _validate_config_types
        good = {"api_port": 8080, "temperature": 0.7, "api_host": "0.0.0.0"}
        cleaned = _validate_config_types(good)
        assert cleaned["api_port"] == 8080
        assert cleaned["temperature"] == 0.7
        assert cleaned["api_host"] == "0.0.0.0"

    def test_unknown_keys_pass_through(self):
        """Keys not in CONFIG defaults should pass through (user extensions)."""
        from enigma_engine.config.defaults import _validate_config_types
        custom = {"my_custom_setting": "whatever"}
        cleaned = _validate_config_types(custom)
        assert cleaned["my_custom_setting"] == "whatever"

