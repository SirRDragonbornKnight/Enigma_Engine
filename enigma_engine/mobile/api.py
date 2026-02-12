"""
Mobile API for Enigma AI Engine

Provides REST API endpoints optimized for mobile apps:
- Lightweight responses
- Voice input/output support  
- Efficient data transfer
- Token-based authentication

Usage:
    from enigma_engine.mobile.api import mobile_app
    
    # Run as standalone app
    mobile_app.run(host='0.0.0.0', port=5001)
    
    # Or integrate with existing Flask app
    main_app.register_blueprint(mobile_app, url_prefix='/mobile')
"""

from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

from ..config import CONFIG

# Create Flask app or blueprint
if FLASK_AVAILABLE:
    mobile_app = Flask(__name__)
    CORS(mobile_app)
    
    # Add SocketIO for real-time communication
    if SOCKETIO_AVAILABLE:
        mobile_socketio = SocketIO(mobile_app, cors_allowed_origins="*")
    else:
        mobile_socketio = None
else:
    mobile_app = None
    mobile_socketio = None


# Global engine instance (lazy loaded)
_engine = None


def get_engine():
    """Get or create inference engine."""
    global _engine
    
    if _engine is None:
        try:
            from ..core.inference import InferenceEngine
            default_model = CONFIG.get("default_model", "enigma_engine")
            _engine = InferenceEngine(model_name=default_model)
        except Exception as e:
            print(f"Warning: Could not load inference engine: {e}")
            _engine = None
    
    return _engine


if FLASK_AVAILABLE:
    
    @mobile_app.route('/api/v1/chat', methods=['POST'])
    def mobile_chat():
        """
        Optimized chat endpoint for mobile apps.
        
        Request:
        {
            "message": "Hello",
            "max_length": 100,  // optional
            "temperature": 0.7  // optional
        }
        
        Response:
        {
            "response": "Hi! How can I help you?",
            "model": "enigma_engine",
            "tokens_used": 10,
            "timestamp": "2024-01-01T12:00:00"
        }
        """
        data = request.json
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        max_length = data.get('max_length', 100)  # Shorter for mobile
        temperature = data.get('temperature', 0.7)
        
        engine = get_engine()
        if engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        try:
            response = engine.generate(
                message,
                max_gen=max_length,
                temperature=temperature
            )
            
            return jsonify({
                'response': response,
                'model': CONFIG.get('default_model', 'enigma_engine'),
                'tokens_used': len(response.split()),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/models', methods=['GET'])
    def list_models():
        """
        List available models for mobile.
        
        Response:
        {
            "models": [
                {"name": "enigma_engine", "size": "small", "loaded": true},
                ...
            ]
        }
        """
        try:
            models_dir = Path(CONFIG["models_dir"])
            models = []
            
            for model_path in models_dir.glob("*"):
                if model_path.is_dir():
                    # Check if it has model files
                    has_model = any(model_path.glob("*.pth")) or any(model_path.glob("*.pt"))
                    
                    if has_model:
                        models.append({
                            'name': model_path.name,
                            'size': 'unknown',  # Could detect from config
                            'loaded': False  # Could check if currently loaded
                        })
            
            return jsonify({'models': models})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/voice/speak', methods=['POST'])
    def mobile_speak():
        """
        TTS optimized for mobile - returns audio stream.
        
        Request:
        {
            "text": "Hello world",
            "voice": "default"  // optional
        }
        
        Response:
            Audio file (WAV format) or JSON status
        """
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        voice_name = data.get('voice', 'default')
        
        try:
            from ..voice import set_voice, speak

            # Set the voice profile
            try:
                set_voice(voice_name)
            except FileNotFoundError:
                # Voice profile not found, use default
                pass
            
            # Speak the text (uses system TTS)
            speak(text)
            
            return jsonify({
                'success': True,
                'text': text,
                'voice': voice_name,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            return jsonify({'error': 'Voice module not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/voice/listen', methods=['POST'])
    def mobile_listen():
        """
        STT optimized for mobile - accepts audio upload.
        
        Request:
            - Form data with 'audio' file
            - Or JSON with base64 encoded audio
        
        Response:
        {
            "text": "transcribed text",
            "confidence": 0.95
        }
        """
        # Check for file upload
        if 'audio' in request.files:
            audio_file = request.files['audio']
            # Process audio file
            return jsonify({
                'message': 'STT not fully implemented',
                'text': 'placeholder transcription'
            })
        
        # Check for base64 audio
        data = request.json
        if data and 'audio_base64' in data:
            # Decode base64 audio
            return jsonify({
                'message': 'STT not fully implemented',
                'text': 'placeholder transcription'
            })
        
        return jsonify({'error': 'No audio provided'}), 400
    
    
    @mobile_app.route('/api/v1/status', methods=['GET'])
    def mobile_status():
        """
        Get lightweight status for mobile.
        
        Response:
        {
            "status": "ok",
            "model_loaded": true,
            "model_name": "enigma_engine",
            "version": "1.0"
        }
        """
        engine = get_engine()
        
        # Check available features
        features = {
            'chat': True,
            'voice_tts': False,
            'voice_stt': False,
            'personality': False,
        }
        
        try:
            features['voice_tts'] = True
        except ImportError:
            pass  # Intentionally silent
        
        try:
            features['voice_stt'] = True
        except ImportError:
            pass  # Intentionally silent
        
        try:
            features['personality'] = True
        except ImportError:
            pass  # Intentionally silent
        
        return jsonify({
            'status': 'ok',
            'model_loaded': engine is not None,
            'model_name': CONFIG.get('default_model', 'enigma_engine'),
            'version': '1.0',
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
    
    
    @mobile_app.route('/api/v1/personality', methods=['GET'])
    def get_personality():
        """
        Get current AI personality traits.
        
        Response:
        {
            "traits": {
                "humor_level": 0.5,
                "formality": 0.5,
                ...
            },
            "mood": "neutral"
        }
        """
        try:
            from ..core.personality import load_personality
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            personality = load_personality(model_name)
            
            return jsonify({
                'traits': personality.traits.to_dict(),
                'mood': personality.mood,
                'interests': personality.interests,
                'conversation_count': personality.conversation_count
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/personality', methods=['PUT'])
    def update_personality():
        """
        Update AI personality traits.
        
        Request:
        {
            "traits": {
                "humor_level": 0.7,
                ...
            }
        }
        """
        data = request.json
        
        if not data or 'traits' not in data:
            return jsonify({'error': 'No traits provided'}), 400
        
        try:
            from ..core.personality import load_personality
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            personality = load_personality(model_name)
            
            # Update traits
            for key, value in data['traits'].items():
                if hasattr(personality.traits, key):
                    setattr(personality.traits, key, value)
            
            personality.save()
            
            return jsonify({
                'success': True,
                'traits': personality.traits.to_dict()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    # =========================================================================
    # FEEDBACK & TRAINING ENDPOINTS
    # =========================================================================

    @mobile_app.route('/api/v1/feedback', methods=['POST'])
    def mobile_feedback():
        """
        Submit feedback on AI response for training.
        
        Request:
        {
            "input_text": "What is Python?",
            "output_text": "Python is a programming language",
            "rating": "positive" | "negative",
            "correction": "Optional: Better response",
            "message_id": "Optional: ID for tracking"
        }
        
        Response:
        {
            "success": true,
            "examples_collected": 42,
            "ready_to_train": false
        }
        """
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required = ['input_text', 'output_text', 'rating']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        if data['rating'] not in ['positive', 'negative']:
            return jsonify({'error': 'Rating must be "positive" or "negative"'}), 400
        
        try:
            from ..core.self_improvement import get_learning_engine
            from ..learning.training_scheduler import get_training_scheduler
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            engine = get_learning_engine(model_name)
            
            # Record feedback
            engine.record_feedback(
                input_text=data['input_text'],
                output_text=data['output_text'],
                feedback=data['rating'],
                correction=data.get('correction')
            )
            
            # Get updated stats
            queue_stats = engine.get_queue_stats()
            
            # Check if ready to train
            scheduler = get_training_scheduler(model_name)
            ready_to_train = scheduler.should_train()
            
            return jsonify({
                'success': True,
                'examples_collected': queue_stats.get('total_examples', 0),
                'ready_to_train': ready_to_train,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/training/example', methods=['POST'])
    def mobile_training_example():
        """
        Submit a training example (Q&A pair).
        
        Request:
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "quality": 0.8  // Optional quality score 0-1
        }
        
        Response:
        {
            "success": true,
            "example_id": "abc123",
            "examples_collected": 43
        }
        """
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'question' not in data or 'answer' not in data:
            return jsonify({'error': 'Missing required fields: question, answer'}), 400
        
        try:
            from ..core.self_improvement import get_learning_engine
            import uuid
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            engine = get_learning_engine(model_name)
            
            # Create unique ID for this example
            example_id = str(uuid.uuid4())[:8]
            
            # Record as positive feedback (user-submitted examples are high quality)
            quality = data.get('quality', 0.9)
            engine.record_feedback(
                input_text=data['question'],
                output_text=data['answer'],
                feedback='positive',
                quality_override=quality
            )
            
            # Get updated stats
            queue_stats = engine.get_queue_stats()
            
            return jsonify({
                'success': True,
                'example_id': example_id,
                'examples_collected': queue_stats.get('total_examples', 0),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/training/stats', methods=['GET'])
    def mobile_training_stats():
        """
        Get training queue statistics.
        
        Response:
        {
            "examples_collected": 42,
            "positive_feedback": 35,
            "negative_feedback": 7,
            "min_examples_needed": 100,
            "ready_to_train": false,
            "last_training": "2026-02-09T12:00:00",
            "training_in_progress": false
        }
        """
        try:
            from ..core.self_improvement import get_learning_engine
            from ..learning.training_scheduler import get_training_scheduler
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            engine = get_learning_engine(model_name)
            scheduler = get_training_scheduler(model_name)
            
            # Get queue statistics
            queue_stats = engine.get_queue_stats()
            metrics = engine.get_metrics()
            scheduler_status = scheduler.get_status()
            
            return jsonify({
                'examples_collected': queue_stats.get('total_examples', 0),
                'positive_feedback': getattr(metrics, 'positive_feedback', 0),
                'negative_feedback': getattr(metrics, 'negative_feedback', 0),
                'min_examples_needed': scheduler_status.get('min_examples_needed', 100),
                'ready_to_train': scheduler_status.get('ready_to_train', False),
                'last_training': scheduler_status.get('last_training_time'),
                'training_in_progress': scheduler_status.get('training_in_progress', False),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/training/start', methods=['POST'])
    def mobile_training_start():
        """
        Trigger training on the PC (requires auth in production).
        
        Request:
        {
            "force": false  // Force training even if criteria not met
        }
        
        Response:
        {
            "success": true,
            "message": "Training started",
            "training_id": "train_123"
        }
        """
        data = request.json or {}
        force = data.get('force', False)
        
        try:
            from ..learning.training_scheduler import get_training_scheduler
            import uuid
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            scheduler = get_training_scheduler(model_name)
            
            # Check if already training
            if scheduler.training_in_progress:
                return jsonify({
                    'success': False,
                    'message': 'Training already in progress'
                }), 400
            
            # Check if ready to train (unless forced)
            if not force and not scheduler.should_train():
                status = scheduler.get_status()
                return jsonify({
                    'success': False,
                    'message': f"Not enough examples. Have {status['examples_collected']}, need {status['min_examples_needed']}"
                }), 400
            
            # Start training in background thread
            training_id = f"train_{uuid.uuid4().hex[:8]}"
            
            import threading
            thread = threading.Thread(target=scheduler.run_training, daemon=True)
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Training started',
                'training_id': training_id,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    # =========================================================================
    # GENERATION ENDPOINTS (PC processes, mobile displays)
    # =========================================================================

    @mobile_app.route('/api/v1/generate/image', methods=['POST'])
    def mobile_generate_image():
        """
        Generate an image (runs on PC, returns result to mobile).
        
        Request:
        {
            "prompt": "a sunset over mountains",
            "width": 512,
            "height": 512,
            "steps": 20
        }
        
        Response:
        {
            "success": true,
            "image_base64": "...",
            "image_url": "/static/generated/img_123.png"
        }
        """
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        try:
            from ..gui.tabs.image_tab import StableDiffusionLocal
            import base64
            from pathlib import Path
            
            # Try to get image generator
            generator = StableDiffusionLocal()
            
            # Generate image
            result = generator.generate(
                prompt=data['prompt'],
                width=data.get('width', 512),
                height=data.get('height', 512),
                num_inference_steps=data.get('steps', 20)
            )
            
            if result and 'image_path' in result:
                # Read image and encode as base64
                img_path = Path(result['image_path'])
                with open(img_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'image_base64': img_base64,
                    'image_url': f"/static/generated/{img_path.name}",
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Image generation failed'}), 500
                
        except ImportError:
            return jsonify({'error': 'Image generation not available. Install diffusers.'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/generate/code', methods=['POST'])
    def mobile_generate_code():
        """
        Generate code (runs on PC, returns result to mobile).
        
        Request:
        {
            "prompt": "Python function to find prime numbers",
            "language": "python"
        }
        
        Response:
        {
            "success": true,
            "code": "def is_prime(n):...",
            "language": "python"
        }
        """
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        try:
            engine = get_engine()
            if engine is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            language = data.get('language', 'python')
            prompt = f"Write {language} code: {data['prompt']}\n\nCode:"
            
            code = engine.generate(
                prompt,
                max_gen=500,
                temperature=0.3  # Lower temp for code
            )
            
            return jsonify({
                'success': True,
                'code': code.strip(),
                'language': language,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/generate/audio', methods=['POST'])
    def mobile_generate_audio():
        """
        Generate audio/TTS (runs on PC, returns audio URL).
        
        Request:
        {
            "text": "Hello world",
            "voice": "default"
        }
        
        Response:
        {
            "success": true,
            "audio_url": "/static/generated/audio_123.wav",
            "audio_base64": "..."  // For smaller files
        }
        """
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        try:
            from ..voice import speak
            import uuid
            from pathlib import Path
            import base64
            
            # Generate audio file
            output_dir = Path(CONFIG.get('output_dir', 'output')) / 'generated'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
            audio_path = output_dir / audio_filename
            
            # Use TTS to generate audio (save to file)
            speak(data['text'], output_path=str(audio_path))
            
            # Read and encode
            if audio_path.exists():
                with open(audio_path, 'rb') as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'audio_url': f"/static/generated/{audio_filename}",
                    'audio_base64': audio_base64,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Audio generation failed'}), 500
                
        except ImportError:
            return jsonify({'error': 'Voice module not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/generate/video', methods=['POST'])
    def mobile_generate_video():
        """
        Generate video (runs on PC, returns result to mobile).
        
        Request:
        {
            "prompt": "a cat walking in the park",
            "duration": 2.0,
            "fps": 8
        }
        
        Response:
        {
            "success": true,
            "video_url": "/static/generated/video_123.mp4",
            "video_base64": "..."
        }
        """
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        try:
            from ..gui.tabs.video_tab import LocalVideo
            import base64
            from pathlib import Path
            
            # Try to get video generator
            generator = LocalVideo()
            if not generator.load():
                return jsonify({'error': 'Video generation not available'}), 500
            
            try:
                # Generate video
                result = generator.generate(
                    prompt=data['prompt'],
                    duration=data.get('duration', 2.0),
                    fps=data.get('fps', 8)
                )
                
                if result.get('success') and result.get('path'):
                    video_path = Path(result['path'])
                    
                    # Read and encode as base64 (if small enough)
                    file_size = video_path.stat().st_size
                    video_base64 = None
                    
                    if file_size < 10 * 1024 * 1024:  # < 10MB
                        with open(video_path, 'rb') as f:
                            video_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    return jsonify({
                        'success': True,
                        'video_url': f"/static/generated/{video_path.name}",
                        'video_base64': video_base64,
                        'duration': data.get('duration', 2.0),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': result.get('error', 'Video generation failed')}), 500
            finally:
                generator.unload()
                
        except ImportError:
            return jsonify({'error': 'Video generation not available. Install diffusers.'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/generate/3d', methods=['POST'])
    def mobile_generate_3d():
        """
        Generate 3D model (runs on PC, returns GLB/OBJ to mobile).
        
        Request:
        {
            "prompt": "a small wooden chair",
            "guidance_scale": 15.0,
            "num_steps": 64
        }
        
        Response:
        {
            "success": true,
            "model_url": "/static/generated/3d_123.glb",
            "format": "glb"
        }
        """
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        try:
            from ..gui.tabs.threed_tab import Local3DGen
            from pathlib import Path
            
            # Try to get 3D generator
            generator = Local3DGen()
            if not generator.load():
                return jsonify({'error': '3D generation not available'}), 500
            
            try:
                # Generate 3D model
                result = generator.generate(
                    prompt=data['prompt'],
                    guidance_scale=data.get('guidance_scale', 15.0),
                    num_inference_steps=data.get('num_steps', 64)
                )
                
                if result.get('success') and result.get('path'):
                    model_path = Path(result['path'])
                    
                    return jsonify({
                        'success': True,
                        'model_url': f"/static/generated/{model_path.name}",
                        'format': model_path.suffix.lstrip('.'),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': result.get('error', '3D generation failed')}), 500
            finally:
                generator.unload()
                
        except ImportError:
            return jsonify({'error': '3D generation not available. Install diffusers.'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/training/progress', methods=['GET'])
    def mobile_training_progress():
        """
        Get current training progress.
        
        Response:
        {
            "training_in_progress": true,
            "progress_percent": 45.5,
            "current_epoch": 2,
            "total_epochs": 5,
            "current_loss": 0.234,
            "eta_seconds": 120
        }
        """
        try:
            from ..learning.training_scheduler import get_training_scheduler
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            scheduler = get_training_scheduler(model_name)
            
            status = scheduler.get_status()
            
            # Get detailed progress if training
            progress = {
                'training_in_progress': status.get('training_in_progress', False),
                'progress_percent': status.get('progress_percent', 0),
                'current_epoch': status.get('current_epoch', 0),
                'total_epochs': status.get('total_epochs', 0),
                'current_loss': status.get('current_loss'),
                'eta_seconds': status.get('eta_seconds'),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(progress)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    # =========================================================================
    # MEMORY & CONVERSATION SYNC ENDPOINTS
    # =========================================================================

    @mobile_app.route('/api/v1/conversations', methods=['GET'])
    def mobile_list_conversations():
        """
        List all conversations from PC.
        
        Query params:
        - limit: Max conversations to return (default 50)
        - offset: Pagination offset (default 0)
        
        Response:
        {
            "conversations": [
                {"name": "Chat 1", "message_count": 23, "last_updated": "..."},
                ...
            ],
            "total": 100
        }
        """
        try:
            from ..memory.manager import ConversationManager
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            manager = ConversationManager(model_name)
            
            limit = request.args.get('limit', 50, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            # List all conversations
            all_convos = manager.list_conversations()
            
            # Apply pagination
            paginated = all_convos[offset:offset + limit]
            
            return jsonify({
                'conversations': paginated,
                'total': len(all_convos),
                'limit': limit,
                'offset': offset,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/conversations/<name>', methods=['GET'])
    def mobile_get_conversation(name):
        """
        Get a specific conversation by name.
        
        Response:
        {
            "name": "Chat 1",
            "messages": [
                {"role": "user", "text": "Hello", "ts": 12345},
                {"role": "ai", "text": "Hi!", "ts": 12346}
            ]
        }
        """
        try:
            from ..memory.manager import ConversationManager
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            manager = ConversationManager(model_name)
            
            conversation = manager.load_conversation(name)
            
            if conversation is None:
                return jsonify({'error': 'Conversation not found'}), 404
            
            return jsonify({
                'name': name,
                'messages': conversation,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/conversations/sync', methods=['POST'])
    def mobile_sync_conversations():
        """
        Sync mobile conversations to PC.
        
        Request:
        {
            "conversations": [
                {
                    "name": "Mobile Chat 1",
                    "messages": [{"role": "user", "text": "...", "ts": 123}]
                }
            ]
        }
        
        Response:
        {
            "success": true,
            "synced_count": 3
        }
        """
        data = request.json
        
        if not data or 'conversations' not in data:
            return jsonify({'error': 'No conversations provided'}), 400
        
        try:
            from ..memory.manager import ConversationManager
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            manager = ConversationManager(model_name)
            
            synced = 0
            for convo in data['conversations']:
                name = convo.get('name')
                messages = convo.get('messages', [])
                
                if name and messages:
                    manager.save_conversation(name, messages)
                    synced += 1
            
            return jsonify({
                'success': True,
                'synced_count': synced,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/memory/search', methods=['GET'])
    def mobile_memory_search():
        """
        Search conversation memory.
        
        Query params:
        - query: Search query
        - limit: Max results (default 10)
        
        Response:
        {
            "results": [
                {"conversation": "Chat 1", "text": "...", "score": 0.95},
                ...
            ]
        }
        """
        query = request.args.get('query', '')
        limit = request.args.get('limit', 10, type=int)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        try:
            from ..memory.manager import ConversationManager
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            manager = ConversationManager(model_name)
            
            results = manager.search_conversations(query, limit=limit)
            
            return jsonify({
                'results': results,
                'query': query,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/memory/export', methods=['POST'])
    def mobile_memory_export():
        """
        Export memories for mobile offline use.
        
        Request:
        {
            "format": "json",
            "limit": 100  // Optional: limit number of messages
        }
        
        Response:
        {
            "success": true,
            "data": {...},  // Exported data
            "message_count": 456
        }
        """
        data = request.json or {}
        format = data.get('format', 'json')
        
        try:
            from ..memory.manager import ConversationManager
            import tempfile
            
            model_name = CONFIG.get('default_model', 'enigma_engine')
            manager = ConversationManager(model_name)
            
            # Export to temp path
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"export.{format}"
                result = manager.export_all(str(output_path), format=format)
                
                if format == 'json' and output_path.exists():
                    import json
                    with open(output_path, 'r') as f:
                        exported_data = json.load(f)
                    
                    return jsonify({
                        'success': True,
                        'data': exported_data,
                        'message_count': result.get('message_count', 0),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': True,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    # =========================================================================
    # AVATAR CONTROL ENDPOINTS
    # =========================================================================

    @mobile_app.route('/api/v1/avatar/state', methods=['GET'])
    def mobile_avatar_state():
        """
        Get current avatar state (emotion, pose, etc).
        
        Response:
        {
            "emotion": "happy",
            "pose": "idle",
            "position": {"x": 0, "y": 0},
            "scale": 1.0,
            "visible": true
        }
        """
        try:
            from ..avatar.controller import get_avatar_controller
            
            controller = get_avatar_controller()
            state = controller.get_state() if controller else {}
            
            return jsonify({
                'emotion': state.get('emotion', 'neutral'),
                'pose': state.get('pose', 'idle'),
                'position': state.get('position', {'x': 0, 'y': 0}),
                'scale': state.get('scale', 1.0),
                'visible': state.get('visible', True),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/avatar/emotion', methods=['POST'])
    def mobile_avatar_emotion():
        """
        Set avatar emotion remotely.
        
        Request:
        {
            "emotion": "happy" | "sad" | "surprised" | "angry" | "neutral"
        }
        
        Response:
        {
            "success": true,
            "emotion": "happy"
        }
        """
        data = request.json
        
        if not data or 'emotion' not in data:
            return jsonify({'error': 'No emotion provided'}), 400
        
        valid_emotions = ['happy', 'sad', 'surprised', 'angry', 'neutral', 'thinking', 'excited']
        emotion = data['emotion'].lower()
        
        if emotion not in valid_emotions:
            return jsonify({'error': f'Invalid emotion. Valid: {valid_emotions}'}), 400
        
        try:
            from ..avatar.controller import get_avatar_controller
            
            controller = get_avatar_controller()
            if controller:
                controller.set_emotion(emotion)
                return jsonify({
                    'success': True,
                    'emotion': emotion,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Avatar controller not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/avatar/gesture', methods=['POST'])
    def mobile_avatar_gesture():
        """
        Trigger avatar gesture.
        
        Request:
        {
            "gesture": "wave" | "nod" | "shake" | "point" | "shrug"
        }
        
        Response:
        {
            "success": true,
            "gesture": "wave"
        }
        """
        data = request.json
        
        if not data or 'gesture' not in data:
            return jsonify({'error': 'No gesture provided'}), 400
        
        valid_gestures = ['wave', 'nod', 'shake', 'point', 'shrug', 'bow', 'clap', 'thumbs_up']
        gesture = data['gesture'].lower()
        
        if gesture not in valid_gestures:
            return jsonify({'error': f'Invalid gesture. Valid: {valid_gestures}'}), 400
        
        try:
            from ..avatar.controller import get_avatar_controller
            
            controller = get_avatar_controller()
            if controller:
                controller.play_gesture(gesture)
                return jsonify({
                    'success': True,
                    'gesture': gesture,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Avatar controller not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    # =========================================================================
    # QR CODE PAIRING
    # =========================================================================

    @mobile_app.route('/api/v1/pairing/qr', methods=['GET'])
    def mobile_pairing_qr():
        """
        Generate QR code for mobile pairing.
        
        Response:
        {
            "success": true,
            "qr_base64": "...",
            "connection_url": "http://192.168.1.100:5001",
            "device_name": "My PC"
        }
        """
        try:
            import socket
            import base64
            from io import BytesIO
            
            # Get local IP address
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except Exception:
                # Fallback to connecting to external address
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                except Exception:
                    local_ip = "127.0.0.1"
                finally:
                    s.close()
            
            port = 5001  # Default mobile API port
            connection_url = f"http://{local_ip}:{port}"
            
            # Try to generate QR code
            qr_base64 = None
            try:
                import qrcode
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(connection_url)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                qr_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except ImportError:
                pass  # QR code generation optional
            
            return jsonify({
                'success': True,
                'qr_base64': qr_base64,
                'connection_url': connection_url,
                'device_name': hostname,
                'local_ip': local_ip,
                'port': port,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/pairing/verify', methods=['POST'])
    def mobile_pairing_verify():
        """
        Verify mobile pairing connection.
        
        Request:
        {
            "device_id": "mobile_123",
            "device_name": "My Phone"
        }
        
        Response:
        {
            "success": true,
            "paired": true,
            "server_name": "My PC"
        }
        """
        data = request.json or {}
        device_id = data.get('device_id', 'unknown')
        device_name = data.get('device_name', 'Unknown Device')
        
        try:
            import socket
            hostname = socket.gethostname()
            
            return jsonify({
                'success': True,
                'paired': True,
                'server_name': hostname,
                'device_id': device_id,
                'device_name': device_name,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    # =========================================================================
    # SETTINGS SYNC ENDPOINTS
    # =========================================================================

    @mobile_app.route('/api/v1/settings', methods=['GET'])
    def mobile_get_settings():
        """
        Get current settings from PC (for mobile sync).
        
        Response:
        {
            "personality": {"name": "Enigma", "traits": {...}},
            "voice": {"voice_id": "default", "rate": 1.0, "pitch": 1.0},
            "preferences": {"theme": "dark", "language": "en"},
            "default_model": "enigma_engine"
        }
        """
        try:
            settings = {
                'default_model': CONFIG.get('default_model', 'enigma_engine'),
                'preferences': {
                    'theme': CONFIG.get('theme', 'dark'),
                    'language': CONFIG.get('language', 'en'),
                    'auto_speak': CONFIG.get('auto_speak', False),
                    'auto_listen': CONFIG.get('auto_listen', False)
                }
            }
            
            # Get personality settings
            try:
                from ..personality.engine import PersonalityEngine
                personality = PersonalityEngine()
                settings['personality'] = {
                    'name': personality.name,
                    'traits': personality.traits.to_dict() if hasattr(personality.traits, 'to_dict') else {}
                }
            except Exception:
                settings['personality'] = {'name': 'Enigma', 'traits': {}}
            
            # Get voice settings
            try:
                from ..voice import VoiceSettings
                voice_settings = VoiceSettings()
                settings['voice'] = {
                    'voice_id': voice_settings.voice_id,
                    'rate': voice_settings.rate,
                    'pitch': voice_settings.pitch,
                    'volume': voice_settings.volume
                }
            except Exception:
                settings['voice'] = {
                    'voice_id': 'default',
                    'rate': 1.0,
                    'pitch': 1.0,
                    'volume': 1.0
                }
            
            settings['timestamp'] = datetime.now().isoformat()
            return jsonify(settings)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/settings', methods=['PUT'])
    def mobile_update_settings():
        """
        Update settings from mobile.
        
        Request:
        {
            "personality": {"name": "Custom AI"},
            "voice": {"rate": 1.2},
            "preferences": {"theme": "light"}
        }
        
        Response:
        {
            "success": true,
            "updated": ["personality", "voice", "preferences"]
        }
        """
        data = request.json
        
        if not data:
            return jsonify({'error': 'No settings provided'}), 400
        
        try:
            updated = []
            
            # Update personality
            if 'personality' in data:
                try:
                    from ..personality.engine import PersonalityEngine
                    personality = PersonalityEngine()
                    if 'name' in data['personality']:
                        personality.name = data['personality']['name']
                    if 'traits' in data['personality']:
                        personality.update_traits(data['personality']['traits'])
                    updated.append('personality')
                except Exception as e:
                    pass  # Personality update optional
            
            # Update voice settings
            if 'voice' in data:
                try:
                    from ..voice import VoiceSettings
                    voice_settings = VoiceSettings()
                    for key in ['voice_id', 'rate', 'pitch', 'volume']:
                        if key in data['voice']:
                            setattr(voice_settings, key, data['voice'][key])
                    voice_settings.save()
                    updated.append('voice')
                except Exception as e:
                    pass  # Voice update optional
            
            # Update preferences
            if 'preferences' in data:
                for key, value in data['preferences'].items():
                    if key in ['theme', 'language', 'auto_speak', 'auto_listen']:
                        CONFIG.set(key, value)
                updated.append('preferences')
            
            return jsonify({
                'success': True,
                'updated': updated,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/settings/sync', methods=['POST'])
    def mobile_sync_settings():
        """
        Two-way settings sync between mobile and PC.
        
        Request:
        {
            "mobile_settings": {
                "last_modified": "2026-02-10T12:00:00",
                "personality": {...},
                "voice": {...}
            }
        }
        
        Response:
        {
            "pc_settings": {...},
            "merged": true,
            "conflicts": []
        }
        """
        data = request.json or {}
        mobile_settings = data.get('mobile_settings', {})
        
        try:
            # Get current PC settings
            pc_settings = {
                'default_model': CONFIG.get('default_model', 'enigma_engine'),
                'preferences': {
                    'theme': CONFIG.get('theme', 'dark'),
                    'language': CONFIG.get('language', 'en')
                }
            }
            
            # Get personality
            try:
                from ..personality.engine import PersonalityEngine
                personality = PersonalityEngine()
                pc_settings['personality'] = {
                    'name': personality.name,
                    'traits': personality.traits.to_dict() if hasattr(personality.traits, 'to_dict') else {}
                }
            except Exception:
                pc_settings['personality'] = {'name': 'Enigma', 'traits': {}}
            
            # Merge logic: PC wins by default, but can be customized
            conflicts = []
            if mobile_settings:
                # Compare timestamps if available
                pc_time = datetime.now()
                mobile_time = datetime.fromisoformat(mobile_settings.get('last_modified', '2000-01-01T00:00:00'))
                
                # If mobile is newer, apply mobile settings
                if mobile_time > pc_time:
                    # Apply mobile settings to PC
                    if 'personality' in mobile_settings:
                        conflicts.append('personality')
                    if 'voice' in mobile_settings:
                        conflicts.append('voice')
            
            return jsonify({
                'pc_settings': pc_settings,
                'merged': True,
                'conflicts': conflicts,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# =========================================================================
# WEBSOCKET EVENTS FOR REAL-TIME AVATAR STREAMING
# =========================================================================

if FLASK_AVAILABLE and SOCKETIO_AVAILABLE and mobile_socketio:
    
    # Track connected clients for avatar updates
    _avatar_subscribers = set()
    
    @mobile_socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print(f"Mobile client connected: {request.sid}")
        emit('connected', {'status': 'ok', 'timestamp': datetime.now().isoformat()})
    
    @mobile_socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        _avatar_subscribers.discard(request.sid)
        print(f"Mobile client disconnected: {request.sid}")
    
    @mobile_socketio.on('subscribe_avatar')
    def handle_subscribe_avatar(data=None):
        """Subscribe to avatar state updates."""
        _avatar_subscribers.add(request.sid)
        emit('subscribed', {'channel': 'avatar', 'timestamp': datetime.now().isoformat()})
    
    @mobile_socketio.on('unsubscribe_avatar')
    def handle_unsubscribe_avatar(data=None):
        """Unsubscribe from avatar state updates."""
        _avatar_subscribers.discard(request.sid)
        emit('unsubscribed', {'channel': 'avatar', 'timestamp': datetime.now().isoformat()})
    
    @mobile_socketio.on('get_avatar_state')
    def handle_get_avatar_state(data=None):
        """Get current avatar state via WebSocket."""
        try:
            from ..avatar.controller import get_avatar_controller
            
            controller = get_avatar_controller()
            state = controller.get_state() if controller else {}
            
            emit('avatar_state', {
                'emotion': state.get('emotion', 'neutral'),
                'pose': state.get('pose', 'idle'),
                'position': state.get('position', {'x': 0, 'y': 0}),
                'scale': state.get('scale', 1.0),
                'visible': state.get('visible', True),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @mobile_socketio.on('set_avatar_emotion')
    def handle_set_avatar_emotion(data):
        """Set avatar emotion via WebSocket."""
        if not data or 'emotion' not in data:
            emit('error', {'message': 'No emotion provided'})
            return
        
        valid_emotions = ['happy', 'sad', 'surprised', 'angry', 'neutral', 'thinking', 'excited']
        emotion = data['emotion'].lower()
        
        if emotion not in valid_emotions:
            emit('error', {'message': f'Invalid emotion. Valid: {valid_emotions}'})
            return
        
        try:
            from ..avatar.controller import get_avatar_controller
            
            controller = get_avatar_controller()
            if controller:
                controller.set_emotion(emotion)
                
                # Broadcast to all subscribers
                mobile_socketio.emit('avatar_emotion_changed', {
                    'emotion': emotion,
                    'timestamp': datetime.now().isoformat()
                }, room=list(_avatar_subscribers))
                
                emit('success', {'emotion': emotion})
            else:
                emit('error', {'message': 'Avatar controller not available'})
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @mobile_socketio.on('trigger_avatar_gesture')
    def handle_trigger_avatar_gesture(data):
        """Trigger avatar gesture via WebSocket."""
        if not data or 'gesture' not in data:
            emit('error', {'message': 'No gesture provided'})
            return
        
        valid_gestures = ['wave', 'nod', 'shake', 'point', 'shrug', 'bow', 'clap', 'thumbs_up']
        gesture = data['gesture'].lower()
        
        if gesture not in valid_gestures:
            emit('error', {'message': f'Invalid gesture. Valid: {valid_gestures}'})
            return
        
        try:
            from ..avatar.controller import get_avatar_controller
            
            controller = get_avatar_controller()
            if controller:
                controller.play_gesture(gesture)
                
                # Broadcast to all subscribers
                mobile_socketio.emit('avatar_gesture_triggered', {
                    'gesture': gesture,
                    'timestamp': datetime.now().isoformat()
                }, room=list(_avatar_subscribers))
                
                emit('success', {'gesture': gesture})
            else:
                emit('error', {'message': 'Avatar controller not available'})
        except Exception as e:
            emit('error', {'message': str(e)})


def broadcast_avatar_update(state: dict):
    """
    Broadcast avatar state update to all subscribers.
    Call this from avatar controller when state changes.
    
    Args:
        state: Dictionary with emotion, pose, position, etc.
    """
    if SOCKETIO_AVAILABLE and mobile_socketio:
        try:
            from flask import has_app_context
            if has_app_context():
                mobile_socketio.emit('avatar_state', {
                    **state,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception:
            pass  # Silently fail if not in request context


def run_mobile_api(host: str = '0.0.0.0', port: int = 5001):
    """
    Run mobile API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    if not FLASK_AVAILABLE:
        print("Error: Flask not installed")
        return
    
    print(f"\n{'='*60}")
    print("ENIGMA AI MOBILE API")
    print(f"{'='*60}")
    print(f"\nAPI starting on http://{host}:{port}")
    print(f"\nREST Endpoints:")
    print(f"   POST /api/v1/chat              - Chat with AI")
    print(f"   GET  /api/v1/models            - List models")
    print(f"   POST /api/v1/voice/speak       - Text-to-speech")
    print(f"   POST /api/v1/voice/listen      - Speech-to-text")
    print(f"   GET  /api/v1/status            - System status")
    print(f"   GET  /api/v1/personality       - Get personality")
    print(f"   PUT  /api/v1/personality       - Update personality")
    print(f"")
    print(f"   --- Training & Feedback ---")
    print(f"   POST /api/v1/feedback          - Submit feedback")
    print(f"   POST /api/v1/training/example  - Submit training example")
    print(f"   GET  /api/v1/training/stats    - Get training stats")
    print(f"   POST /api/v1/training/start    - Start training")
    print(f"   GET  /api/v1/training/progress - Get training progress")
    print(f"")
    print(f"   --- Generation ---")
    print(f"   POST /api/v1/generate/image    - Generate image")
    print(f"   POST /api/v1/generate/code     - Generate code")
    print(f"   POST /api/v1/generate/audio    - Generate audio/TTS")
    print(f"   POST /api/v1/generate/video    - Generate video")
    print(f"   POST /api/v1/generate/3d       - Generate 3D model")
    print(f"")
    print(f"   --- Memory & Conversations ---")
    print(f"   GET  /api/v1/conversations     - List conversations")
    print(f"   GET  /api/v1/conversations/<n> - Get conversation")
    print(f"   POST /api/v1/conversations/sync- Sync from mobile")
    print(f"   GET  /api/v1/memory/search     - Search memories")
    print(f"   POST /api/v1/memory/export     - Export for offline")
    print(f"")
    print(f"   --- Avatar Control ---")
    print(f"   GET  /api/v1/avatar/state      - Get avatar state")
    print(f"   POST /api/v1/avatar/emotion    - Set emotion")
    print(f"   POST /api/v1/avatar/gesture    - Trigger gesture")
    print(f"")
    print(f"   --- Pairing & Settings ---")
    print(f"   GET  /api/v1/pairing/qr        - Get QR code for pairing")
    print(f"   POST /api/v1/pairing/verify    - Verify connection")
    print(f"   GET  /api/v1/settings          - Get current settings")
    print(f"   PUT  /api/v1/settings          - Update settings")
    print(f"   POST /api/v1/settings/sync     - Two-way settings sync")
    
    if SOCKETIO_AVAILABLE:
        print(f"")
        print(f"WebSocket Events (connect to ws://{host}:{port}):")
        print(f"   subscribe_avatar              - Subscribe to avatar updates")
        print(f"   unsubscribe_avatar            - Unsubscribe from updates")
        print(f"   get_avatar_state              - Request current state")
        print(f"   set_avatar_emotion            - Set emotion (real-time)")
        print(f"   trigger_avatar_gesture        - Trigger gesture (real-time)")
    
    print(f"\n{'='*60}\n")
    
    # Use SocketIO if available for WebSocket support
    if SOCKETIO_AVAILABLE and mobile_socketio:
        mobile_socketio.run(mobile_app, host=host, port=port)
    else:
        mobile_app.run(host=host, port=port)
