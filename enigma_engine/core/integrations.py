"""
================================================================================
            TRAINER AI INTEGRATIONS - CONNECTING ALL SYSTEMS
================================================================================

Integration layer that connects TrainerAI to all existing Enigma systems:
- CharacterTrainer (enigma_engine/tools/data_trainer.py) → Character extraction
- TaskTrainer (enigma_engine/tools/data_trainer.py) → Task data generation
- train_specialized_model.py → Model training
- ToolRouter (enigma_engine/core/tool_router.py) → Route to new models
- ModelRegistry (enigma_engine/core/model_registry.py) → Store models
- ContextTracker (enigma_engine/utils/context_window.py) → Token tracking
- ConversationManager (enigma_engine/memory/manager.py) → Save/load conversations

📍 FILE: enigma_engine/core/integrations.py
🏷️ TYPE: Integration Layer
🎯 MAIN CLASS: TrainerIntegration

USAGE:
    from enigma_engine.core.integrations import TrainerIntegration, get_integration
    
    integration = get_integration()
    
    # Create AI from character with full pipeline
    result = integration.create_ai_from_character(
        character_name="Sherlock Holmes",
        data_path="data/training/",
        train_model=True,  # Actually train the model
        register_with_router=True  # Make available in ToolRouter
    )
    
    # Generate and train task-specific AI
    result = integration.create_task_ai(
        task_type="code",
        model_size="small",
        train=True
    )
    
    # Track token usage during generation
    with integration.track_context() as tracker:
        data = integration.trainer.generate_training_data("chat", count=100)
        print(f"Generated data used {tracker.get_usage().used_tokens} tokens")

SEE ALSO:
    - enigma_engine/core/trainer_ai.py - TrainerAI class
    - enigma_engine/tools/data_trainer.py - CharacterTrainer, TaskTrainer
    - enigma_engine/core/tool_router.py - ToolRouter
    - enigma_engine/core/model_registry.py - ModelRegistry
    - enigma_engine/utils/context_window.py - ContextTracker
    - enigma_engine/memory/manager.py - ConversationManager
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TYPE_CHECKING

from ..config import CONFIG

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from .trainer_ai import TrainerAI
    from ..tools.data_trainer import CharacterTrainer, TaskTrainer
    from .tool_router import ToolRouter
    from .model_registry import ModelRegistry
    from ..utils.context_window import ContextTracker
    from ..memory.manager import ConversationManager

logger = logging.getLogger(__name__)


# =============================================================================
# INTEGRATION RESULT TYPES
# =============================================================================

@dataclass
class IntegrationResult:
    """Result from an integration operation."""
    success: bool
    operation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class TrainingProgress:
    """Progress information for training operations."""
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    best_loss: float = float('inf')
    status: str = "not_started"  # not_started, training, completed, failed
    message: str = ""


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class TrainerIntegration:
    """
    Integration layer connecting TrainerAI to all Enigma systems.
    
    Provides unified methods for:
    - Character-to-AI pipeline with actual model training
    - Task-specific AI creation
    - Model registration with ToolRouter
    - Token tracking during generation
    - Conversation history management
    """
    
    def __init__(
        self,
        trainer_ai: Optional["TrainerAI"] = None,
        model_registry: Optional["ModelRegistry"] = None,
        tool_router: Optional["ToolRouter"] = None,
        context_tracker: Optional["ContextTracker"] = None,
        conversation_manager: Optional["ConversationManager"] = None,
    ):
        """
        Initialize the integration layer.
        
        Args:
            trainer_ai: TrainerAI instance (created if not provided)
            model_registry: ModelRegistry instance (created if not provided)
            tool_router: ToolRouter instance (created if not provided)
            context_tracker: ContextTracker for token tracking
            conversation_manager: ConversationManager for history
        """
        self._trainer_ai = trainer_ai
        self._model_registry = model_registry
        self._tool_router = tool_router
        self._context_tracker = context_tracker
        self._conversation_manager = conversation_manager
        
        # Training progress tracking
        self._training_progress: Optional[TrainingProgress] = None
        self._training_callbacks: List[Callable[[TrainingProgress], None]] = []
        
        logger.info("TrainerIntegration initialized")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAZY PROPERTY ACCESSORS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def trainer(self) -> "TrainerAI":
        """Get or create TrainerAI instance."""
        if self._trainer_ai is None:
            from .trainer_ai import get_trainer_ai
            self._trainer_ai = get_trainer_ai()
        return self._trainer_ai
    
    @property
    def model_registry(self) -> "ModelRegistry":
        """Get or create ModelRegistry instance."""
        if self._model_registry is None:
            from .model_registry import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    @property
    def tool_router(self) -> "ToolRouter":
        """Get or create ToolRouter instance."""
        if self._tool_router is None:
            from .tool_router import get_router
            self._tool_router = get_router()
        return self._tool_router
    
    @property
    def context_tracker(self) -> "ContextTracker":
        """Get or create ContextTracker instance."""
        if self._context_tracker is None:
            from ..utils.context_window import get_context_tracker
            self._context_tracker = get_context_tracker()
        return self._context_tracker
    
    @property
    def conversation_manager(self) -> "ConversationManager":
        """Get or create ConversationManager instance."""
        if self._conversation_manager is None:
            from ..memory.manager import ConversationManager
            self._conversation_manager = ConversationManager()
        return self._conversation_manager
    
    def get_character_trainer(self) -> "CharacterTrainer":
        """Get a CharacterTrainer instance."""
        from ..tools.data_trainer import CharacterTrainer
        return CharacterTrainer()
    
    def get_task_trainer(self, examples_dir: Optional[str] = None) -> "TaskTrainer":
        """Get a TaskTrainer instance."""
        from ..tools.data_trainer import TaskTrainer
        return TaskTrainer(examples_dir=examples_dir)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHARACTER-TO-AI INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_ai_from_character(
        self,
        character_name: str,
        data_path: str,
        output_name: Optional[str] = None,
        model_size: str = "small",
        training_count: int = 500,
        personality_mode: str = "hybrid",
        train_model: bool = True,
        epochs: Optional[int] = None,
        register_with_router: bool = True,
        create_bundle: bool = True,
        aliases: Optional[List[str]] = None,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> IntegrationResult:
        """
        Full Character-to-AI pipeline with model training and registration.
        
        This is the COMPLETE pipeline:
        1. Extract character using CharacterTrainer
        2. Generate training data using TrainerAI
        3. Train a specialized chat model
        4. Register the model with ModelRegistry
        5. Add routing for the new model in ToolRouter
        6. Create an AI bundle
        
        Args:
            character_name: Name of character to extract
            data_path: Path to source training data
            output_name: Name for the AI (defaults to character name)
            model_size: Model size for training (nano, tiny, small, medium, etc.)
            training_count: Number of training examples to generate
            personality_mode: How to apply personality (baked, system_prompt, hybrid)
            train_model: If True, actually train the model
            epochs: Training epochs (None = use default for chat)
            register_with_router: If True, register trained model with ToolRouter
            create_bundle: If True, create an .enigma-bundle
            aliases: Alternative names for the character
            on_progress: Callback for training progress updates
        
        Returns:
            IntegrationResult with paths and status
        """
        result = IntegrationResult(
            success=False,
            operation="create_ai_from_character",
            data={
                "character_name": character_name,
                "model_size": model_size,
                "train_model": train_model,
            }
        )
        
        output_name = output_name or character_name
        safe_name = output_name.lower().replace(" ", "_")
        
        try:
            # Step 1: Use TrainerAI's character_to_ai for extraction and data generation
            logger.info(f"Step 1: Extracting character and generating training data...")
            char_result = self.trainer.character_to_ai(
                character_name=character_name,
                data_path=data_path,
                output_name=output_name,
                training_count=training_count,
                personality_mode=personality_mode,
                create_bundle=False,  # We'll create bundle later with model
                aliases=aliases,
            )
            
            if not char_result.get("success"):
                result.errors.append(char_result.get("error", "Character extraction failed"))
                return result
            
            result.data["character_profile"] = char_result.get("character_profile")
            result.data["training_data_path"] = char_result.get("training_data_path")
            result.data["system_prompt"] = char_result.get("system_prompt")
            
            training_data_path = Path(char_result["training_data_path"])
            
            # Step 2: Train the model if requested
            model_path = None
            if train_model:
                logger.info(f"Step 2: Training {model_size} model...")
                
                # Register progress callback
                if on_progress:
                    self._training_callbacks.append(on_progress)
                
                try:
                    model_path = self._train_specialized_model(
                        model_type="chat",
                        data_path=training_data_path,
                        model_size=model_size,
                        epochs=epochs,
                        output_name=f"{safe_name}_chat",
                    )
                    
                    if model_path:
                        result.data["model_path"] = str(model_path)
                        logger.info(f"Model trained and saved to: {model_path}")
                    else:
                        result.warnings.append("Model training completed but path not returned")
                        
                finally:
                    if on_progress:
                        self._training_callbacks.remove(on_progress)
            else:
                result.data["model_path"] = None
                result.warnings.append("Model training skipped (train_model=False)")
            
            # Step 3: Register with ModelRegistry
            if model_path:
                logger.info(f"Step 3: Registering model in ModelRegistry...")
                try:
                    self.model_registry.create_model(
                        name=safe_name,
                        size=model_size,
                        description=f"Character AI based on {character_name}",
                    )
                    result.data["registry_name"] = safe_name
                except Exception as e:
                    result.warnings.append(f"ModelRegistry registration warning: {e}")
            
            # Step 4: Register with ToolRouter
            if register_with_router and model_path:
                logger.info(f"Step 4: Registering with ToolRouter...")
                try:
                    self._register_model_with_router(
                        tool_name="chat",
                        model_id=f"forge:{safe_name}",
                        model_path=model_path,
                        priority=5,
                    )
                    result.data["router_registered"] = True
                except Exception as e:
                    result.warnings.append(f"ToolRouter registration warning: {e}")
                    result.data["router_registered"] = False
            
            # Step 5: Create bundle
            if create_bundle:
                logger.info(f"Step 5: Creating AI bundle...")
                model_paths = {}
                if model_path:
                    model_paths["chat"] = model_path
                
                bundle_path = self.trainer.create_bundle(
                    name=output_name,
                    description=f"AI based on {character_name} character",
                    model_paths=model_paths,
                    persona_name=output_name,
                    personality=self._summarize_profile(char_result.get("character_profile", {})),
                    system_prompt=char_result.get("system_prompt", "") if personality_mode in ["system_prompt", "hybrid"] else "",
                    tools_enabled=["chat"],
                )
                result.data["bundle_path"] = str(bundle_path)
                
                # Save character profile in bundle
                if char_result.get("character_profile"):
                    profile_path = bundle_path / "character_profile.json"
                    with open(profile_path, 'w', encoding='utf-8') as f:
                        json.dump(char_result["character_profile"], f, indent=2)
            
            result.success = True
            logger.info(f"Successfully created AI '{output_name}' from character '{character_name}'")
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"create_ai_from_character failed: {e}", exc_info=True)
        
        return result
    
    def _summarize_profile(self, profile_dict: Dict[str, Any]) -> str:
        """Create a brief personality summary from profile dict."""
        if not profile_dict:
            return ""
        
        name = profile_dict.get("name", "Character")
        traits = profile_dict.get("personality_traits", {})
        
        strong_traits = [
            trait for trait, score in traits.items()
            if isinstance(score, (int, float)) and score > 0.3
        ]
        
        if strong_traits:
            return f"{name}: {', '.join(strong_traits)}"
        return f"{name} character"
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK-SPECIFIC AI INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_task_ai(
        self,
        task_type: str,
        model_size: Optional[str] = None,
        data_count: int = 200,
        custom_examples_path: Optional[str] = None,
        train: bool = True,
        epochs: Optional[int] = None,
        register_with_router: bool = True,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> IntegrationResult:
        """
        Create a task-specific AI with training.
        
        Uses TaskTrainer and TrainerAI to:
        1. Generate or load task-specific training data
        2. Train a specialized model
        3. Register with ToolRouter
        
        Args:
            task_type: Type of task (image, avatar, tools, code, router, vision, math)
            model_size: Model size (None = use recommended for task)
            data_count: Number of training examples to generate
            custom_examples_path: Path to custom example JSON file
            train: If True, train the model
            epochs: Training epochs (None = use default)
            register_with_router: Register trained model with router
            on_progress: Callback for progress updates
        
        Returns:
            IntegrationResult with training info and paths
        """
        result = IntegrationResult(
            success=False,
            operation="create_task_ai",
            data={"task_type": task_type}
        )
        
        try:
            # Determine model size from TrainerAI's position configs
            position_config = self.trainer.get_position_info(task_type)
            if model_size is None:
                if position_config:
                    model_size = position_config.recommended_model_size
                else:
                    model_size = "small"
            
            result.data["model_size"] = model_size
            
            # Step 1: Generate training data
            logger.info(f"Step 1: Generating training data for {task_type}...")
            
            # Try TaskTrainer first for specific task types
            task_trainer_types = ["image", "avatar", "tools", "code", "web", "file"]
            
            if task_type in task_trainer_types:
                # Use TaskTrainer for these types
                task_trainer = self.get_task_trainer()
                
                # Load custom examples if provided
                if custom_examples_path:
                    task_trainer.load_examples_from_file(custom_examples_path)
                
                # Generate training file
                data_dir = Path(CONFIG.get("data_dir", "data")) / "specialized"
                data_dir.mkdir(parents=True, exist_ok=True)
                training_file = data_dir / f"{task_type}_training.txt"
                
                # Use TaskTrainer's generation methods
                if task_type == "image":
                    task_trainer.generate_image_training(str(training_file))
                elif task_type == "avatar":
                    task_trainer.generate_avatar_training(str(training_file))
                elif task_type == "tools":
                    task_trainer.generate_tool_training(str(training_file))
                elif task_type == "code":
                    task_trainer.generate_code_training(str(training_file))
                else:
                    # Generate using TrainerAI for other types
                    training_data = self.trainer.generate_training_data(task_type, count=data_count)
                    self.trainer.save_training_data(task_type, training_data)
                    training_file = data_dir / f"{task_type}_training.txt"
            else:
                # Use TrainerAI's generate_training_data for router positions
                training_data = self.trainer.generate_training_data(task_type, count=data_count)
                training_file = self.trainer.save_training_data(task_type, training_data)
            
            result.data["training_data_path"] = str(training_file)
            
            # Step 2: Train the model
            model_path = None
            if train:
                logger.info(f"Step 2: Training {model_size} {task_type} model...")
                
                if on_progress:
                    self._training_callbacks.append(on_progress)
                
                try:
                    model_path = self._train_specialized_model(
                        model_type=task_type,
                        data_path=training_file,
                        model_size=model_size,
                        epochs=epochs,
                    )
                    
                    if model_path:
                        result.data["model_path"] = str(model_path)
                finally:
                    if on_progress:
                        self._training_callbacks.remove(on_progress)
            
            # Step 3: Register with router
            if register_with_router and model_path:
                logger.info(f"Step 3: Registering with ToolRouter...")
                try:
                    # Map task types to router tool names
                    tool_mapping = {
                        "image": "image",
                        "avatar": "avatar",
                        "code": "code",
                        "router": "router",
                        "vision": "vision",
                        "math": "math",
                        "chat": "chat",
                    }
                    
                    tool_name = tool_mapping.get(task_type, task_type)
                    
                    self._register_model_with_router(
                        tool_name=tool_name,
                        model_id=f"forge:{task_type}",
                        model_path=model_path,
                        priority=10,  # Higher priority for specialized models
                    )
                    result.data["router_registered"] = True
                except Exception as e:
                    result.warnings.append(f"Router registration warning: {e}")
            
            result.success = True
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"create_task_ai failed: {e}", exc_info=True)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODEL TRAINING INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _train_specialized_model(
        self,
        model_type: str,
        data_path: Path,
        model_size: str,
        epochs: Optional[int] = None,
        output_name: Optional[str] = None,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
    ) -> Optional[Path]:
        """
        Train a specialized model using the train_specialized_model script logic.
        
        This integrates with scripts/train_specialized_model.py functionality
        but can be called programmatically.
        
        Args:
            model_type: Type of model (router, vision, code, chat, avatar, etc.)
            data_path: Path to training data
            model_size: Model size preset
            epochs: Training epochs
            output_name: Custom output name
            batch_size: Training batch size
            learning_rate: Learning rate
        
        Returns:
            Path to trained model or None if failed
        """
        try:
            import torch
            from .model import create_model
            from .tokenizer import get_tokenizer
            from .training import Trainer, TrainingConfig
        except ImportError as e:
            logger.error(f"Cannot import training dependencies: {e}")
            return None
        
        data_path = Path(data_path)
        if not data_path.exists():
            logger.error(f"Training data not found: {data_path}")
            return None
        
        # Load training data
        training_text = data_path.read_text(encoding='utf-8')
        
        # Default epochs based on model type
        default_epochs = {
            "router": 50,
            "vision": 40,
            "code": 40,
            "math": 40,
            "trainer": 60,
            "avatar": 50,
            "chat": 40,
        }
        
        if epochs is None:
            epochs = default_epochs.get(model_type, 40)
        
        # Get tokenizer
        try:
            vocab_file = Path(CONFIG.get("models_dir", "models")) / "vocab" / "bpe_vocab.json"
            if vocab_file.exists():
                tokenizer = get_tokenizer("bpe", vocab_path=str(vocab_file))
            else:
                tokenizer = get_tokenizer("char")
        except Exception:
            tokenizer = get_tokenizer("char")
        
        # Create model
        logger.info(f"Creating {model_size} model for {model_type}...")
        model = create_model(size=model_size, vocab_size=tokenizer.vocab_size)
        
        # Determine output path
        output_dir = Path(CONFIG.get("models_dir", "models")) / "specialized"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_name:
            output_path = output_dir / f"{output_name}.pth"
        else:
            output_path = output_dir / f"{model_type}_{model_size}.pth"
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training config
        checkpoint_dir = output_dir / "checkpoints" / f"{model_type}_{model_size}"
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=True,
            save_every=max(1, epochs // 5),
            checkpoint_dir=str(checkpoint_dir),
        )
        
        # Create trainer
        trainer = Trainer(model, tokenizer, device=device, config=config)
        
        # Update progress tracking
        self._training_progress = TrainingProgress(
            total_epochs=epochs,
            status="training",
            message=f"Training {model_type} model..."
        )
        
        def on_epoch(progress_dict: Dict[str, Any]):
            """Callback for epoch completion (receives dict with epoch, loss, lr)."""
            if self._training_progress:
                self._training_progress.current_epoch = progress_dict.get('epoch', 0)
                self._training_progress.current_loss = progress_dict.get('loss', 0.0)
                if self._training_progress.current_loss < self._training_progress.best_loss:
                    self._training_progress.best_loss = self._training_progress.current_loss
                
                # Notify callbacks
                for callback in self._training_callbacks:
                    try:
                        callback(self._training_progress)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
        
        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        try:
            # Wrap in list if string (trainer.train expects list[str])
            texts = [training_text] if isinstance(training_text, str) else training_text
            trainer.train(texts, callback=on_epoch)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.get_config(),
                'model_type': model_type,
                'model_size': model_size,
                'vocab_size': tokenizer.vocab_size,
                'epochs': epochs,
            }, output_path)
            
            # Save config JSON
            config_path = output_path.with_suffix('.json')
            with open(config_path, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'model_size': model_size,
                    'vocab_size': tokenizer.vocab_size,
                    'config': model.get_config(),
                    'epochs': epochs,
                }, f, indent=2)
            
            self._training_progress.status = "completed"
            self._training_progress.message = f"Training complete: {output_path}"
            
            logger.info(f"Model saved to: {output_path}")
            return output_path
            
        except KeyboardInterrupt:
            logger.info("Training interrupted")
            self._training_progress.status = "failed"
            self._training_progress.message = "Training interrupted by user"
            return None
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._training_progress.status = "failed"
            self._training_progress.message = str(e)
            raise
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOOL ROUTER INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _register_model_with_router(
        self,
        tool_name: str,
        model_id: str,
        model_path: Path,
        priority: int = 5,
        model_type: str = "forge",
    ) -> bool:
        """
        Register a trained model with the ToolRouter.
        
        Args:
            tool_name: Name of the tool/route (e.g., "chat", "code", "image")
            model_id: Unique model identifier (e.g., "forge:my_model")
            model_path: Path to model weights
            priority: Routing priority (higher = preferred)
            model_type: Type of model ("forge", "huggingface", "local", "api")
        
        Returns:
            True if successfully registered
        """
        try:
            # Use the correct API: assign_model(tool_name, model_id, priority, config)
            self.tool_router.assign_model(
                tool_name=tool_name,
                model_id=model_id,
                priority=priority,
                config={"path": str(model_path), "model_type": model_type}
            )
            logger.info(f"Registered {model_id} with router for '{tool_name}' (priority={priority})")
            
            # Note: assign_model already calls _save_config() internally
            
            return True
        except Exception as e:
            logger.error(f"Failed to register model with router: {e}")
            return False
    
    def list_router_assignments(self, tool_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        List current model assignments in the router.
        
        Args:
            tool_name: Specific tool to list, or None for all
        
        Returns:
            Dict mapping tool names to their model assignments
        """
        assignments = {}
        
        # Use routing_rules (not rules) and get_assignments (not get_models_for_tool)
        for name in self.tool_router.routing_rules.keys():
            if tool_name and name != tool_name:
                continue
            
            models = self.tool_router.get_assignments(name)
            assignments[name] = [
                {
                    "model_id": m.model_id,
                    "model_type": m.model_type,
                    "priority": m.priority,
                    "config": m.config,
                }
                for m in models
            ]
        
        return assignments
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT TRACKING INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    @contextmanager
    def track_context(
        self,
        max_tokens: int = 4096,
        on_warning: Optional[Callable[[float], None]] = None,
    ) -> Generator["ContextTracker", None, None]:
        """
        Context manager for tracking token usage during operations.
        
        Usage:
            with integration.track_context() as tracker:
                data = integration.trainer.generate_training_data("chat", count=100)
                print(f"Used {tracker.get_usage().percentage:.1f}% of context")
        
        Args:
            max_tokens: Maximum context window size
            on_warning: Callback when usage exceeds warning threshold
        
        Yields:
            ContextTracker instance
        """
        from ..utils.context_window import ContextTracker, ContextConfig
        
        config = ContextConfig(max_tokens=max_tokens)
        tracker = ContextTracker(config=config)
        
        if on_warning:
            def check_warning(usage):
                if usage.percentage >= config.warn_percentage:
                    on_warning(usage.percentage)
            tracker.add_usage_callback(check_warning)
        
        try:
            yield tracker
        finally:
            pass  # Cleanup if needed
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated token count
        """
        # estimate_tokens returns TokenEstimate object, extract the count
        estimate = self.context_tracker.estimate_tokens(text)
        return estimate.estimated_tokens
    
    def get_context_usage(self) -> Dict[str, Any]:
        """
        Get current context usage statistics.
        
        Returns:
            Dict with usage information
        """
        usage = self.context_tracker.get_usage()
        return {
            "used_tokens": usage.used_tokens,
            "max_tokens": usage.max_tokens,
            "remaining_tokens": usage.remaining_tokens,
            "percentage": usage.percentage,
            "level": usage.level.name,
            "message_count": usage.message_count,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONVERSATION MANAGER INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_training_conversation(
        self,
        name: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save a training conversation to history.
        
        Useful for saving the conversation used to generate training data
        or to track AI creation sessions.
        
        Args:
            name: Conversation name
            messages: List of message dicts
            metadata: Optional metadata to include
        
        Returns:
            True if saved successfully
        """
        try:
            # Add metadata to conversation
            full_messages = list(messages)
            if metadata:
                full_messages.insert(0, {
                    "role": "system",
                    "text": f"Metadata: {json.dumps(metadata)}",
                    "ts": datetime.now().timestamp(),
                })
            
            self.conversation_manager.save_conversation(name, full_messages)
            logger.info(f"Saved training conversation: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False
    
    def load_training_conversation(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load a training conversation from history.
        
        Args:
            name: Conversation name
        
        Returns:
            List of messages or None if not found
        """
        try:
            data = self.conversation_manager.load_conversation(name)
            if data:
                return data.get("messages", [])
            return None
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return None
    
    def list_training_conversations(self) -> List[str]:
        """
        List all saved training conversations.
        
        Returns:
            List of conversation names
        """
        return self.conversation_manager.list_conversations()
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search conversations by text content.
        
        Args:
            query: Search query (text to match)
            limit: Maximum results
        
        Returns:
            List of matching conversation snippets
        """
        # ConversationManager doesn't have text search - implement manually
        results = []
        query_lower = query.lower()
        
        for conv_name in self.conversation_manager.list_conversations():
            try:
                data = self.conversation_manager.load_conversation(conv_name)
                messages = data.get("messages", [])
                
                # Search through messages
                for msg in messages:
                    text = msg.get("text", "")
                    if query_lower in text.lower():
                        results.append({
                            "conversation": conv_name,
                            "role": msg.get("role", "unknown"),
                            "text": text[:200] + "..." if len(text) > 200 else text,
                            "timestamp": msg.get("ts"),
                        })
                        if len(results) >= limit:
                            return results
            except Exception:
                continue
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # BUNDLE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def load_and_activate_bundle(
        self,
        bundle_path: Path,
        set_as_default: bool = False,
    ) -> IntegrationResult:
        """
        Load an AI bundle and activate it in the system.
        
        This loads the bundle's models into ModelRegistry and
        registers them with ToolRouter.
        
        Args:
            bundle_path: Path to bundle directory
            set_as_default: If True, set as default for included tools
        
        Returns:
            IntegrationResult with activation status
        """
        result = IntegrationResult(
            success=False,
            operation="load_and_activate_bundle",
            data={"bundle_path": str(bundle_path)}
        )
        
        try:
            # Load bundle spec
            bundle_spec = self.trainer.load_bundle(bundle_path)
            result.data["bundle_name"] = bundle_spec.name
            result.data["positions"] = bundle_spec.positions_trained
            
            # Load each model
            bundle_path = Path(bundle_path)
            for position, model_rel_path in bundle_spec.models.items():
                model_path = bundle_path / model_rel_path
                
                if model_path.exists():
                    # Register with ModelRegistry
                    model_name = f"{bundle_spec.name.lower()}_{position}"
                    try:
                        self.model_registry.create_model(
                            name=model_name,
                            description=f"{bundle_spec.name} - {position}",
                        )
                    except Exception:
                        pass  # Model may already exist
                    
                    # Register with ToolRouter
                    priority = 15 if set_as_default else 5
                    self._register_model_with_router(
                        tool_name=position,
                        model_id=f"forge:{model_name}",
                        model_path=model_path,
                        priority=priority,
                    )
                    
                    logger.info(f"Activated {position} model from bundle")
                else:
                    result.warnings.append(f"Model not found: {model_rel_path}")
            
            result.success = True
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Bundle activation failed: {e}")
        
        return result


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_integration_instance: Optional[TrainerIntegration] = None


def get_integration() -> TrainerIntegration:
    """
    Get the global TrainerIntegration instance.
    
    Returns:
        TrainerIntegration singleton
    """
    global _integration_instance
    
    if _integration_instance is None:
        _integration_instance = TrainerIntegration()
    
    return _integration_instance


def reset_integration() -> None:
    """Reset the global integration instance."""
    global _integration_instance
    _integration_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_create_character_ai(
    character_name: str,
    data_path: str,
    model_size: str = "small",
) -> IntegrationResult:
    """
    Quick helper to create an AI from a character.
    
    Args:
        character_name: Name of character
        data_path: Path to training data
        model_size: Model size to use
    
    Returns:
        IntegrationResult
    """
    return get_integration().create_ai_from_character(
        character_name=character_name,
        data_path=data_path,
        model_size=model_size,
    )


def quick_create_task_ai(
    task_type: str,
    model_size: Optional[str] = None,
) -> IntegrationResult:
    """
    Quick helper to create a task-specific AI.
    
    Args:
        task_type: Type of task (code, image, avatar, etc.)
        model_size: Model size (None = auto)
    
    Returns:
        IntegrationResult
    """
    return get_integration().create_task_ai(
        task_type=task_type,
        model_size=model_size,
    )
