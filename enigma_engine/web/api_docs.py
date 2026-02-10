"""
API Documentation Generator
===========================

Auto-generate OpenAPI/Swagger documentation for the Enigma Engine API.

Features:
- Extracts routes from Flask app
- Generates OpenAPI 3.0 spec
- Serves Swagger UI
- Exports to JSON/YAML

Usage:
    from enigma_engine.web.api_docs import generate_api_docs, serve_swagger
    
    # Generate OpenAPI spec
    spec = generate_api_docs()
    
    # Save to file
    save_openapi_spec('openapi.json')
    
    # Add Swagger UI to Flask app
    add_swagger_ui(app)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# OpenAPI 3.0 base structure
OPENAPI_BASE = {
    "openapi": "3.0.0",
    "info": {
        "title": "Enigma Engine API",
        "description": "AI-powered conversational engine with text generation, personality management, voice synthesis, and more.",
        "version": "1.0.0",
        "contact": {
            "name": "Enigma Engine",
            "url": "https://github.com/enigma-engine/enigma-engine"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    "servers": [
        {
            "url": "http://localhost:8080",
            "description": "Local development server"
        }
    ],
    "tags": [
        {"name": "Status", "description": "Server status and health checks"},
        {"name": "Generation", "description": "Text generation endpoints"},
        {"name": "Models", "description": "Model management"},
        {"name": "Personality", "description": "AI personality configuration"},
        {"name": "Voice", "description": "Voice and speech settings"},
        {"name": "Memory", "description": "Conversation history and memory"},
        {"name": "Training", "description": "Model training endpoints"},
        {"name": "Push", "description": "Push notification management"},
        {"name": "Pages", "description": "Web interface pages"},
    ],
    "paths": {},
    "components": {
        "schemas": {},
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer"
            },
            "apiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }
        }
    }
}


# Predefined schemas for common types
COMMON_SCHEMAS = {
    "GenerateRequest": {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Input text prompt",
                "example": "Hello, how are you?"
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens to generate",
                "default": 200,
                "minimum": 1,
                "maximum": 4096
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 2.0
            },
            "stream": {
                "type": "boolean",
                "description": "Enable streaming response",
                "default": False
            }
        }
    },
    "GenerateResponse": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Generated text"
            },
            "tokens_generated": {
                "type": "integer",
                "description": "Number of tokens generated"
            },
            "generation_time": {
                "type": "number",
                "description": "Time taken in seconds"
            }
        }
    },
    "PersonalityConfig": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "AI personality name"
            },
            "traits": {
                "type": "object",
                "description": "Personality traits",
                "properties": {
                    "friendliness": {"type": "number", "minimum": 0, "maximum": 1},
                    "formality": {"type": "number", "minimum": 0, "maximum": 1},
                    "humor": {"type": "number", "minimum": 0, "maximum": 1},
                    "verbosity": {"type": "number", "minimum": 0, "maximum": 1},
                    "creativity": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "voice_preferences": {
                "type": "object",
                "description": "Voice synthesis preferences"
            }
        }
    },
    "VoiceConfig": {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "Whether voice is enabled"
            },
            "voice_id": {
                "type": "string",
                "description": "Voice identifier"
            },
            "speed": {
                "type": "number",
                "description": "Speech rate multiplier",
                "default": 1.0
            },
            "pitch": {
                "type": "number",
                "description": "Voice pitch adjustment"
            }
        }
    },
    "StatusResponse": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ready", "loading", "error"]
            },
            "model_loaded": {
                "type": "boolean"
            },
            "model_name": {
                "type": "string"
            },
            "uptime": {
                "type": "number"
            },
            "version": {
                "type": "string"
            }
        }
    },
    "Error": {
        "type": "object",
        "properties": {
            "error": {
                "type": "string",
                "description": "Error message"
            },
            "code": {
                "type": "integer",
                "description": "Error code"
            }
        }
    },
    "PushSubscription": {
        "type": "object",
        "required": ["endpoint", "keys"],
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "Push service endpoint URL"
            },
            "keys": {
                "type": "object",
                "properties": {
                    "p256dh": {"type": "string"},
                    "auth": {"type": "string"}
                }
            }
        }
    },
    "ModelInfo": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "size": {
                "type": "string"
            },
            "parameters": {
                "type": "integer"
            },
            "loaded": {
                "type": "boolean"
            }
        }
    }
}


# Route documentation hints (manual overrides for complex routes)
ROUTE_HINTS = {
    '/api/status': {
        'summary': 'Get server status',
        'description': 'Returns current server status, model information, and health metrics.',
        'tags': ['Status'],
        'responses': {
            '200': {'description': 'Server status', 'schema': 'StatusResponse'}
        }
    },
    '/api/generate': {
        'summary': 'Generate text',
        'description': 'Generate text completion from a prompt using the loaded model.',
        'tags': ['Generation'],
        'request_body': 'GenerateRequest',
        'responses': {
            '200': {'description': 'Generated response', 'schema': 'GenerateResponse'},
            '400': {'description': 'Invalid request', 'schema': 'Error'},
            '500': {'description': 'Generation error', 'schema': 'Error'}
        }
    },
    '/api/models': {
        'summary': 'List available models',
        'description': 'Get list of available models and their status.',
        'tags': ['Models'],
        'responses': {
            '200': {'description': 'List of models', 'schema': {'type': 'array', 'items': {'$ref': '#/components/schemas/ModelInfo'}}}
        }
    },
    '/api/personality': {
        'GET': {
            'summary': 'Get personality config',
            'description': 'Get current AI personality configuration.',
            'tags': ['Personality'],
            'responses': {
                '200': {'description': 'Personality config', 'schema': 'PersonalityConfig'}
            }
        },
        'POST': {
            'summary': 'Update personality',
            'description': 'Update AI personality traits and settings.',
            'tags': ['Personality'],
            'request_body': 'PersonalityConfig',
            'responses': {
                '200': {'description': 'Updated config', 'schema': 'PersonalityConfig'}
            }
        }
    },
    '/api/personality/reset': {
        'summary': 'Reset personality',
        'description': 'Reset personality to default settings.',
        'tags': ['Personality'],
        'responses': {
            '200': {'description': 'Reset successful'}
        }
    },
    '/api/voice': {
        'GET': {
            'summary': 'Get voice config',
            'description': 'Get current voice synthesis settings.',
            'tags': ['Voice'],
            'responses': {
                '200': {'description': 'Voice config', 'schema': 'VoiceConfig'}
            }
        },
        'POST': {
            'summary': 'Update voice settings',
            'description': 'Update voice synthesis settings.',
            'tags': ['Voice'],
            'request_body': 'VoiceConfig',
            'responses': {
                '200': {'description': 'Updated config', 'schema': 'VoiceConfig'}
            }
        }
    },
    '/api/voice/preview': {
        'summary': 'Preview voice',
        'description': 'Generate a voice preview with current settings.',
        'tags': ['Voice'],
        'responses': {
            '200': {'description': 'Audio preview URL'}
        }
    },
    '/api/voice/profiles': {
        'summary': 'List voice profiles',
        'description': 'Get available voice profiles.',
        'tags': ['Voice'],
        'responses': {
            '200': {'description': 'List of voice profiles'}
        }
    },
    '/api/voice/transcribe': {
        'summary': 'Transcribe audio',
        'description': 'Transcribe audio to text using speech recognition.',
        'tags': ['Voice'],
        'responses': {
            '200': {'description': 'Transcribed text'}
        }
    },
    '/api/memory': {
        'GET': {
            'summary': 'Get conversation history',
            'description': 'Get recent conversation history.',
            'tags': ['Memory'],
            'responses': {
                '200': {'description': 'Conversation history'}
            }
        },
        'DELETE': {
            'summary': 'Clear memory',
            'description': 'Clear conversation history.',
            'tags': ['Memory'],
            'responses': {
                '200': {'description': 'Memory cleared'}
            }
        }
    },
    '/api/push/vapid-key': {
        'summary': 'Get VAPID public key',
        'description': 'Get the VAPID public key for push subscriptions.',
        'tags': ['Push'],
        'responses': {
            '200': {'description': 'VAPID public key'}
        }
    },
    '/api/push/subscribe': {
        'summary': 'Subscribe to push notifications',
        'description': 'Register a push subscription endpoint.',
        'tags': ['Push'],
        'request_body': 'PushSubscription',
        'responses': {
            '200': {'description': 'Subscription registered'}
        }
    },
    '/api/push/unsubscribe': {
        'summary': 'Unsubscribe from push',
        'description': 'Remove a push subscription.',
        'tags': ['Push'],
        'responses': {
            '200': {'description': 'Unsubscribed'}
        }
    },
    '/api/push/test': {
        'summary': 'Send test notification',
        'description': 'Send a test push notification.',
        'tags': ['Push'],
        'responses': {
            '200': {'description': 'Test notification sent'}
        }
    },
}


def extract_routes_from_app(app) -> List[Dict[str, Any]]:
    """
    Extract route information from a Flask app.
    
    Args:
        app: Flask application instance
        
    Returns:
        List of route dictionaries with path, methods, and metadata
    """
    routes = []
    
    for rule in app.url_map.iter_rules():
        # Skip static files
        if rule.endpoint == 'static':
            continue
        
        # Get view function
        view_func = app.view_functions.get(rule.endpoint)
        
        # Extract docstring
        docstring = ""
        if view_func and view_func.__doc__:
            docstring = view_func.__doc__.strip()
        
        # Parse methods
        methods = [m for m in rule.methods if m not in ('HEAD', 'OPTIONS')]
        
        # Extract path parameters
        path_params = []
        for converter, _args, variable in rule._trace:
            if converter:
                path_params.append({
                    'name': variable,
                    'in': 'path',
                    'required': True,
                    'schema': {'type': 'string'}
                })
        
        routes.append({
            'path': str(rule.rule),
            'methods': methods,
            'endpoint': rule.endpoint,
            'docstring': docstring,
            'path_params': path_params
        })
    
    return routes


def path_to_openapi_format(flask_path: str) -> str:
    """Convert Flask path format to OpenAPI format."""
    # Convert <param> to {param}
    return re.sub(r'<(?:\w+:)?(\w+)>', r'{\1}', flask_path)


def infer_tag_from_path(path: str) -> str:
    """Infer API tag from path."""
    if path.startswith('/api/'):
        parts = path.split('/')
        if len(parts) >= 3:
            tag = parts[2].title()
            return tag
    if path == '/' or path in ('/dashboard', '/chat', '/train', '/settings', '/personality', '/voice', '/memory'):
        return 'Pages'
    return 'Other'


def generate_operation(
    route: Dict[str, Any],
    method: str,
    hints: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate OpenAPI operation object for a route+method.
    
    Args:
        route: Route information dict
        method: HTTP method
        hints: Optional manual hints for this route
        
    Returns:
        OpenAPI operation object
    """
    path = route['path']
    
    # Get hints for this method
    method_hints = {}
    if hints:
        if method in hints:
            method_hints = hints[method]
        elif 'summary' in hints:
            method_hints = hints
    
    # Build operation
    operation = {
        'summary': method_hints.get('summary', f"{method} {path}"),
        'description': method_hints.get('description', route.get('docstring', '')),
        'tags': method_hints.get('tags', [infer_tag_from_path(path)]),
        'operationId': f"{method.lower()}_{route['endpoint'].replace('.', '_')}",
        'responses': {
            '200': {
                'description': 'Successful response',
                'content': {
                    'application/json': {
                        'schema': {'type': 'object'}
                    }
                }
            }
        }
    }
    
    # Add path parameters
    if route.get('path_params'):
        operation['parameters'] = route['path_params']
    
    # Add request body for POST/PUT/PATCH
    if method in ('POST', 'PUT', 'PATCH'):
        request_schema = method_hints.get('request_body')
        if request_schema:
            if isinstance(request_schema, str):
                schema_ref = {'$ref': f'#/components/schemas/{request_schema}'}
            else:
                schema_ref = request_schema
            
            operation['requestBody'] = {
                'required': True,
                'content': {
                    'application/json': {
                        'schema': schema_ref
                    }
                }
            }
    
    # Add custom responses
    if 'responses' in method_hints:
        for code, resp_info in method_hints['responses'].items():
            resp = {'description': resp_info.get('description', '')}
            if 'schema' in resp_info:
                schema = resp_info['schema']
                if isinstance(schema, str):
                    schema = {'$ref': f'#/components/schemas/{schema}'}
                resp['content'] = {'application/json': {'schema': schema}}
            operation['responses'][str(code)] = resp
    
    return operation


def generate_api_docs(app=None) -> Dict[str, Any]:
    """
    Generate OpenAPI specification from Flask app.
    
    Args:
        app: Flask application (uses default if None)
        
    Returns:
        OpenAPI 3.0 specification dict
    """
    # Get Flask app
    if app is None:
        try:
            from . import app as flask_app
            app = flask_app.app
        except ImportError:
            logger.error("Flask app not available")
            return OPENAPI_BASE.copy()
    
    if app is None:
        return OPENAPI_BASE.copy()
    
    # Start with base spec
    spec = json.loads(json.dumps(OPENAPI_BASE))  # Deep copy
    
    # Add common schemas
    spec['components']['schemas'] = COMMON_SCHEMAS.copy()
    
    # Extract routes
    routes = extract_routes_from_app(app)
    
    # Build paths
    for route in routes:
        openapi_path = path_to_openapi_format(route['path'])
        
        # Get hints for this path
        hints = ROUTE_HINTS.get(route['path'], {})
        
        # Initialize path item if needed
        if openapi_path not in spec['paths']:
            spec['paths'][openapi_path] = {}
        
        # Add operations for each method
        for method in route['methods']:
            method_lower = method.lower()
            operation = generate_operation(route, method, hints)
            spec['paths'][openapi_path][method_lower] = operation
    
    return spec


def save_openapi_spec(
    output_path: str = 'openapi.json',
    format: str = 'json',
    app=None
) -> bool:
    """
    Save OpenAPI spec to file.
    
    Args:
        output_path: Output file path
        format: 'json' or 'yaml'
        app: Flask application
        
    Returns:
        True if successful
    """
    spec = generate_api_docs(app)
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                logger.warning("PyYAML not installed, using JSON format")
                with open(output_path, 'w') as f:
                    json.dump(spec, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                json.dump(spec, f, indent=2)
        
        logger.info(f"OpenAPI spec saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save OpenAPI spec: {e}")
        return False


def add_swagger_ui(app, prefix: str = '/docs') -> None:
    """
    Add Swagger UI routes to Flask app.
    
    Args:
        app: Flask application
        prefix: URL prefix for docs (default: /docs)
    """
    from flask import Response
    
    # Swagger UI HTML
    swagger_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Enigma Engine API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css">
    <style>
        body {{ margin: 0; padding: 0; }}
        .swagger-ui .topbar {{ display: none; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({{
            url: "{prefix}/openapi.json",
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
            layout: "BaseLayout",
            deepLinking: true,
            showExtensions: true,
            showCommonExtensions: true
        }});
    </script>
</body>
</html>
'''
    
    @app.route(f'{prefix}')
    @app.route(f'{prefix}/')
    def swagger_ui():
        """Swagger UI documentation interface."""
        return swagger_html
    
    @app.route(f'{prefix}/openapi.json')
    def openapi_spec():
        """OpenAPI specification endpoint."""
        spec = generate_api_docs(app)
        return Response(
            json.dumps(spec, indent=2),
            mimetype='application/json'
        )
    
    @app.route(f'{prefix}/redoc')
    def redoc():
        """ReDoc documentation interface."""
        redoc_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Enigma Engine API Docs - ReDoc</title>
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>body {{ margin: 0; padding: 0; }}</style>
</head>
<body>
    <redoc spec-url="{prefix}/openapi.json"></redoc>
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>
'''
        return redoc_html
    
    logger.info(f"Swagger UI available at {prefix}")
    logger.info(f"ReDoc available at {prefix}/redoc")
    logger.info(f"OpenAPI spec at {prefix}/openapi.json")


def setup_api_docs(app=None) -> None:
    """
    Set up API documentation for the Flask app.
    
    Call this during app initialization to enable docs.
    
    Usage:
        from enigma_engine.web.api_docs import setup_api_docs
        setup_api_docs()  # Uses default app
        
        # Or with custom app
        setup_api_docs(my_app)
    """
    if app is None:
        try:
            from . import app as flask_module
            app = flask_module.app
        except ImportError:
            logger.error("Flask app not available")
            return
    
    if app is not None:
        add_swagger_ui(app)
        logger.info("API documentation enabled")


# Auto-setup when module is imported and Flask is available
def _auto_setup():
    """Auto-setup docs if Flask app exists."""
    try:
        from . import app as flask_module
        if flask_module.app is not None:
            add_swagger_ui(flask_module.app)
    except Exception:
        pass  # Silently fail if app not ready


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate API documentation')
    parser.add_argument('--output', '-o', default='openapi.json',
                        help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json',
                        help='Output format')
    parser.add_argument('--print', '-p', action='store_true',
                        help='Print to stdout')
    
    args = parser.parse_args()
    
    spec = generate_api_docs()
    
    if args.print:
        print(json.dumps(spec, indent=2))
    else:
        save_openapi_spec(args.output, args.format)
        print(f"Saved to {args.output}")
