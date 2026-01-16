"""
Mobile Package for ForgeAI

This module re-exports from forge_ai.comms.mobile_api for backwards compatibility.

The main mobile API implementation lives in forge_ai.comms.mobile_api.

Usage:
    from forge_ai.mobile import MobileAPI, run_mobile_api
    
    # Run API server
    run_mobile_api(host='0.0.0.0', port=5001)
    
    # Or create instance with custom settings
    api = MobileAPI(port=5001, model_name='forge_ai')
    api.run()

Features:
    - Lightweight REST API optimized for mobile apps
    - Conversation context per device
    - Voice input/output (TTS/STT)
    - Flutter and React Native client templates
"""

# Re-export from comms.mobile_api (the main implementation)
from ..comms.mobile_api import (
    MobileAPI,
    create_mobile_api,
    print_mobile_client_template,
    MOBILE_CLIENT_TEMPLATES,
    HAS_FLASK,
)

# Backwards-compatible alias
mobile_app = None
if HAS_FLASK:
    # Create a default instance for simple usage
    _default_api = MobileAPI(port=5001)
    mobile_app = _default_api.app


def run_mobile_api(host: str = '0.0.0.0', port: int = 5001, model_name: str = None):
    """
    Run mobile API server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to bind to (default: 5001)
        model_name: Optional model name to load
    """
    api = MobileAPI(port=port, model_name=model_name)
    api.run(host=host)


__all__ = [
    'MobileAPI',
    'create_mobile_api',
    'run_mobile_api',
    'mobile_app',
    'print_mobile_client_template',
    'MOBILE_CLIENT_TEMPLATES',
]
