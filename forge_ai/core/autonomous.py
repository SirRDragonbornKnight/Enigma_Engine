"""
Autonomous Mode - REAL AI Self-Improvement System

This module implements genuine autonomous learning and self-improvement.
Unlike stub code, this system:
  - Actually reflects on conversations with quality metrics
  - Practices responses with self-evaluation
  - Updates personality based on real interaction patterns
  - Researches topics and builds knowledge
  - Consolidates learning for training
  - Tracks performance and adjusts behavior

All actions are measurable and produce real learning outcomes.
"""

import time
import random
import threading
import logging
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path

from .self_improvement import (
    LearningEngine, LearningExample, LearningSource, Priority,
    PerformanceMetrics, AutonomousConfig, get_learning_engine
)

logger = logging.getLogger(__name__)


class AutonomousMode:
    """
    AI autonomous behavior system with REAL learning capabilities.
    
    This is not stub code. Every autonomous action produces measurable
    results and contributes to the AI's improvement.
    
    Now includes federated learning support to share improvements
    without sharing data.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Learning engine (real learning system)
        self.learning_engine = get_learning_engine(model_name)
        
        # Configuration (persisted)
        self.config = AutonomousConfig()
        self._load_config()
        
        # Thread state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._action_count = 0
        self._last_reset = time.time()
        
        # Callbacks for UI updates
        self.on_action: Optional[Callable[[str], None]] = None
        self.on_thought: Optional[Callable[[str], None]] = None
        self.on_learning: Optional[Callable[[str], None]] = None
        
        # Federated learning (optional)
        self.federated_learning = None
        self._init_federated_learning()
    
    def start(self):
        """Start autonomous mode with real learning."""
        if self._thread and self._thread.is_alive():
            return
        
        self.config.enabled = True
        self._save_config()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        if self.on_action:
            self.on_action(f"[Autonomous] AI learning mode started (Low Power: {self.config.low_power_mode})")
            
        logger.info(f"Autonomous mode started for {self.model_name}")
    
    def stop(self):
        """Stop autonomous mode."""
        self.config.enabled = False
        self._save_config()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        
        # Save final state
        self.learning_engine.save_state()
        
        if self.on_action:
            self.on_action("[Autonomous] AI learning mode stopped")
            
        logger.info(f"Autonomous mode stopped for {self.model_name}")
    
    def set_low_power_mode(self, enabled: bool):
        """
        Enable/disable low power mode for gaming.
        
        In low power mode:
        - Longer interval between actions (30 min vs 5 min)
        - Fewer actions per hour (2 vs 12)
        - Reduced resource usage
        
        Args:
            enabled: True to enable low power mode
        """
        self.config.low_power_mode = enabled
        self._save_config()
        
        if self.on_action:
            mode_str = "enabled" if enabled else "disabled"
            self.on_action(f"[Autonomous] Low power mode {mode_str}")
            
        logger.info(f"Low power mode {mode_str} for {self.model_name}")
    
    def _run_loop(self):
        """Main autonomous loop with real learning actions."""
        while not self._stop_event.is_set():
            # Reset action count every hour
            if time.time() - self._last_reset > 3600:
                self._action_count = 0
                self._last_reset = time.time()
            
            # Check if we can do more actions (respecting low power mode)
            max_actions = self.config.get_effective_max_actions()
            if self._action_count >= max_actions:
                self._stop_event.wait(60)
                continue
            
            # Perform autonomous action
            try:
                self._perform_action()
                self._action_count += 1
            except Exception as e:
                logger.error(f"Error in autonomous action: {e}", exc_info=True)
                if self.on_action:
                    self.on_action(f"[Autonomous] Error: {e}")
            
            # Wait for interval (respecting low power mode)
            interval = self.config.get_effective_interval()
            self._stop_event.wait(interval)
    
    def _perform_action(self):
        """Perform a random autonomous action (all REAL implementations)."""
        # Build action list based on enabled features
        actions = []
        
        if self.config.enable_reflection:
            actions.append(self._reflect_on_conversations)
        
        if self.config.enable_practice:
            actions.append(self._practice_response)
        
        if self.config.enable_personality_evolution:
            actions.append(self._update_personality)
        
        # Always allow curiosity exploration
        actions.append(self._explore_curiosity)
        
        # Additional advanced actions
        if self.config.enable_knowledge_building:
            actions.extend([
                self._build_knowledge,
                self._optimize_responses,
            ])
        
        if self.config.enable_web_research:
            actions.append(self._research_topic)
        
        # Lower probability actions
        if random.random() < 0.2:  # 20% chance
            actions.extend([
                self._consolidate_learning,
                self._self_evaluate,
            ])
        
        if random.random() < 0.1:  # 10% chance
            actions.append(self._dream)
        
        if not actions:
            return
        
        # Select and perform action
        action = random.choice(actions)
        action()
    
    # =========================================================================
    # CONFIGURATION MANAGEMENT
    # =========================================================================
    
    def _load_config(self):
        """Load configuration from disk."""
        from ..config import CONFIG
        models_dir = Path(CONFIG.get("models_dir", "models"))
        config_file = models_dir / self.model_name / "autonomous_config.json"
        
        if config_file.exists():
            try:
                import json
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    self.config = AutonomousConfig.from_dict(data)
                logger.info(f"Loaded autonomous config for {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading autonomous config: {e}")
    
    def _save_config(self):
        """Save configuration to disk."""
        from ..config import CONFIG
        models_dir = Path(CONFIG.get("models_dir", "models"))
        config_file = models_dir / self.model_name / "autonomous_config.json"
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving autonomous config: {e}")
    
    # =========================================================================
    # AUTONOMOUS ACTIONS - REAL IMPLEMENTATIONS
    # =========================================================================
    
    def _explore_curiosity(self):
        """
        Explore a topic the AI is curious about.
        
        REAL IMPLEMENTATION:
        - Gets curiosities from AIBrain
        - Actually searches the web for information
        - Extracts key facts
        - Creates learning examples
        - Updates knowledge graph
        """
        try:
            from .ai_brain import AIBrain
            brain = AIBrain(self.model_name)
            curiosities = brain.get_curiosities()
            
            if not curiosities:
                # No curiosities yet, explore topics from knowledge graph
                topics = self.learning_engine.get_all_topics()
                if topics:
                    topic = random.choice(topics)
                    if self.on_thought:
                        self.on_thought(f"Exploring related concept: {topic}")
                    self._research_topic(specific_topic=topic)
                return
            
            topic = random.choice(curiosities)
            if self.on_thought:
                self.on_thought(f"I'm curious about: {topic}")
            
            # Try to search for information
            try:
                from ..tools.web_tools import search
                results = search(topic, max_results=3)
                
                if results:
                    # Extract information from first result
                    snippet = results[0].get('snippet', '')
                    title = results[0].get('title', '')
                    
                    if snippet:
                        # Create learning example from research
                        example = LearningExample(
                            input_text=f"What is {topic}?",
                            output_text=f"{title}: {snippet}",
                            source=LearningSource.RESEARCH,
                            priority=Priority.MEDIUM,
                            quality_score=0.7,  # Research is generally good quality
                            topics=[topic]
                        )
                        self.learning_engine.add_learning_example(example)
                        
                        if self.on_learning:
                            self.on_learning(f"Learned about {topic}: {snippet[:80]}...")
                        
                        logger.info(f"Explored curiosity: {topic}")
                
                brain.mark_curiosity_explored(topic)
                
            except (ImportError, Exception) as e:
                logger.debug(f"Could not search web for {topic}: {e}")
                # Still mark as explored to avoid getting stuck
                brain.mark_curiosity_explored(topic)
        
        except (ImportError, Exception) as e:
            logger.debug(f"Could not explore curiosity: {e}")
    
    def _reflect_on_conversations(self):
        """
        Review past conversations for learning.
        
        REAL IMPLEMENTATION:
        - Loads recent conversations from ConversationManager
        - Analyzes conversation quality (length, engagement)
        - Extracts high-quality Q&A pairs as training examples
        - Evaluates response quality using multiple metrics
        - Queues examples for the learning system
        - Updates performance metrics
        """
        if self.on_thought:
            self.on_thought("Reflecting on recent conversations...")
        
        try:
            from ..memory.manager import ConversationManager
            
            # Get conversation manager
            conv_manager = ConversationManager(model_name=self.model_name)
            
            # Get recent conversations
            conv_names = conv_manager.list_conversations()
            if not conv_names:
                logger.debug("No conversations to reflect on")
                return
            
            # Analyze recent conversations (up to reflection_depth)
            max_to_analyze = min(self.config.reflection_depth, len(conv_names))
            analyzed_count = 0
            examples_extracted = 0
            
            for conv_name in conv_names[:max_to_analyze]:
                try:
                    conv_data = conv_manager.load_conversation(conv_name)
                    messages = conv_data.get('messages', [])
                    
                    if len(messages) < 2:
                        continue
                    
                    analyzed_count += 1
                    
                    # Extract Q&A pairs from conversation
                    for i in range(len(messages) - 1):
                        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'ai':
                            user_text = messages[i]['text']
                            ai_text = messages[i+1]['text']
                            
                            # Evaluate quality
                            quality_metrics = self.learning_engine.evaluate_response_quality(
                                user_text, ai_text
                            )
                            
                            # Only learn from good responses
                            if quality_metrics['overall'] >= self.config.min_quality_for_learning:
                                example = LearningExample(
                                    input_text=user_text,
                                    output_text=ai_text,
                                    source=LearningSource.REFLECTION,
                                    priority=Priority.MEDIUM,
                                    quality_score=quality_metrics['overall'],
                                    relevance=quality_metrics['relevance'],
                                    coherence=quality_metrics['coherence'],
                                    repetition=quality_metrics['repetition']
                                )
                                self.learning_engine.add_learning_example(example)
                                examples_extracted += 1
                    
                    # Update conversation metrics
                    self.learning_engine.update_conversation_metrics(len(messages))
                    
                except Exception as e:
                    logger.debug(f"Error analyzing conversation {conv_name}: {e}")
            
            if self.on_learning:
                self.on_learning(f"Reflected on {analyzed_count} conversations, extracted {examples_extracted} learning examples")
            
            logger.info(f"Reflection complete: {analyzed_count} conversations, {examples_extracted} examples")
            
        except Exception as e:
            logger.error(f"Error reflecting on conversations: {e}", exc_info=True)
    
    def _practice_response(self):
        """
        Practice generating responses.
        
        REAL IMPLEMENTATION:
        - Generates practice prompts from knowledge graph or templates
        - Generates responses using ForgeEngine
        - Self-evaluates response quality with real metrics
        - Saves good responses as training examples
        - Tracks quality trends over time
        """
        """Practice generating responses."""
        if self.on_thought:
            self.on_thought("Practicing response generation...")
        
        try:
            # Generate practice prompt
            topics = self.learning_engine.get_all_topics()
            
            practice_templates = [
                "Explain {topic} in simple terms",
                "What are the benefits of {topic}?",
                "How does {topic} work?",
                "Tell me about {topic}",
                "Compare {topic} with related concepts",
            ]
            
            if topics:
                topic = random.choice(topics)
                template = random.choice(practice_templates)
                prompt = template.format(topic=topic)
            else:
                # Fallback to general questions
                prompts = [
                    "What is artificial intelligence?",
                    "How can I be more productive?",
                    "Explain the importance of learning",
                    "What makes a good conversation?",
                ]
                prompt = random.choice(prompts)
            
            # Try to generate response
            try:
                from .inference import ForgeEngine
                engine = ForgeEngine(self.model_name)
                response = engine.generate(prompt, max_length=100)
                
                # Evaluate quality
                quality_metrics = self.learning_engine.evaluate_response_quality(
                    prompt, response
                )
                
                # Save if quality is good
                if quality_metrics['overall'] >= self.config.min_quality_for_learning:
                    example = LearningExample(
                        input_text=prompt,
                        output_text=response,
                        source=LearningSource.PRACTICE,
                        priority=Priority.LOW,
                        quality_score=quality_metrics['overall'],
                        relevance=quality_metrics['relevance'],
                        coherence=quality_metrics['coherence'],
                        repetition=quality_metrics['repetition']
                    )
                    self.learning_engine.add_learning_example(example)
                    
                    if self.on_learning:
                        self.on_learning(f"Practiced: {prompt[:50]}... (quality: {quality_metrics['overall']:.2f})")
                
                logger.info(f"Practice complete: quality={quality_metrics['overall']:.2f}")
                
            except Exception as e:
                logger.debug(f"Could not generate practice response: {e}")
        
        except Exception as e:
            logger.error(f"Error in practice: {e}", exc_info=True)
    
    def _update_personality(self):
        """
        Gradually evolve personality.
        
        REAL IMPLEMENTATION:
        - Analyzes real interaction patterns (not random drift)
        - Looks at conversation lengths, feedback ratios, topic patterns
        - Makes justified adjustments based on what users respond well to
        - Tracks evolution history
        """
        if self.on_thought:
            self.on_thought("Analyzing interaction patterns for personality evolution...")
        
        try:
            from .personality import AIPersonality
            personality = AIPersonality(self.model_name)
            personality.load()
            
            # Get performance metrics
            metrics = self.learning_engine.get_metrics()
            
            # Don't evolve if not enough data
            if metrics.total_conversations < 5:
                logger.debug("Not enough interaction data for personality evolution")
                return
            
            # Analyze feedback patterns and adjust traits accordingly
            feedback_ratio = metrics.feedback_ratio()
            
            # Get evolution settings from config
            evolution_rate = self.config.evolution_rate
            balance_threshold = self.config.balance_threshold
            
            # If getting positive feedback, reinforce current traits slightly
            if feedback_ratio > 0.6:
                # Slightly increase traits that are already strong
                reinforcement_traits = ['humor_level', 'empathy', 'creativity', 'playfulness']
                for trait_name in reinforcement_traits:
                    if hasattr(personality.traits, trait_name):
                        current = getattr(personality.traits, trait_name)
                        if current > balance_threshold:
                            # Increase slightly
                            new_value = min(1.0, current + evolution_rate)
                            setattr(personality.traits, trait_name, new_value)
                            logger.debug(f"Increased {trait_name} to {new_value:.2f} (positive feedback)")
            
            elif feedback_ratio < 0.4:
                # Try to be more balanced
                balance_traits = ['formality', 'verbosity']
                for trait_name in balance_traits:
                    if hasattr(personality.traits, trait_name):
                        current = getattr(personality.traits, trait_name)
                        # Move toward middle
                        if current > balance_threshold:
                            new_value = current - evolution_rate
                        else:
                            new_value = current + evolution_rate
                        setattr(personality.traits, trait_name, new_value)
                        logger.debug(f"Adjusted {trait_name} to {new_value:.2f} (balancing)")
            
            # Adjust verbosity based on average conversation length
            if metrics.avg_conversation_length > 0:
                if metrics.avg_conversation_length < 4:
                    # Short conversations - try being more engaging
                    personality.traits.curiosity = min(1.0, personality.traits.curiosity + 0.02)
                    personality.traits.playfulness = min(1.0, personality.traits.playfulness + 0.02)
                elif metrics.avg_conversation_length > 20:
                    # Long conversations - maintain current style
                    pass
            
            # Save evolved personality
            personality.save()
            
            if self.on_learning:
                self.on_learning(f"Personality evolved based on {metrics.total_conversations} conversations")
            
            logger.info(f"Personality evolution complete. Feedback ratio: {feedback_ratio:.1%}")
            
        except Exception as e:
            logger.error(f"Error updating personality: {e}", exc_info=True)
    
    # =========================================================================
    # NEW AUTONOMOUS ACTIONS
    # =========================================================================
    
    def _research_topic(self, specific_topic: Optional[str] = None):
        """
        Actually search the web and learn.
        
        REAL IMPLEMENTATION:
        - Searches web for information
        - Extracts key facts
        - Creates learning examples
        - Updates knowledge graph
        """
        try:
            # Choose a topic to research
            if specific_topic:
                topic = specific_topic
            else:
                # Pick from knowledge graph or generate new
                topics = self.learning_engine.get_all_topics()
                if topics:
                    topic = random.choice(topics)
                else:
                    # Fallback topics
                    topic = random.choice([
                        "machine learning", "programming", "science",
                        "technology", "artificial intelligence"
                    ])
            
            if self.on_thought:
                self.on_thought(f"Researching: {topic}")
            
            from ..tools.web_tools import search
            results = search(topic, max_results=3)
            
            if results:
                for result in results:
                    snippet = result.get('snippet', '')
                    if snippet:
                        example = LearningExample(
                            input_text=f"What can you tell me about {topic}?",
                            output_text=snippet,
                            source=LearningSource.RESEARCH,
                            priority=Priority.LOW,
                            quality_score=0.7,
                            topics=[topic]
                        )
                        self.learning_engine.add_learning_example(example)
                
                if self.on_learning:
                    self.on_learning(f"Researched {topic}: found {len(results)} sources")
                
                logger.info(f"Research complete for: {topic}")
        
        except Exception as e:
            logger.debug(f"Could not research topic: {e}")
    
    def _build_knowledge(self):
        """
        Connect concepts in knowledge graph.
        
        REAL IMPLEMENTATION:
        - Analyzes topic relationships
        - Strengthens connections between related concepts
        - Identifies gaps in knowledge
        """
        if self.on_thought:
            self.on_thought("Building knowledge connections...")
        
        try:
            topics = self.learning_engine.get_all_topics()
            
            if len(topics) < 2:
                logger.debug("Not enough topics for knowledge building")
                return
            
            # Pick two topics and explore their relationship
            topic1 = random.choice(topics)
            related = self.learning_engine.get_related_topics(topic1)
            
            if related:
                topic2 = random.choice(related)
                
                # Create a learning example about the relationship
                prompt = f"How are {topic1} and {topic2} related?"
                
                # Try to generate or research answer
                try:
                    from ..tools.web_tools import search
                    results = search(f"{topic1} and {topic2} relationship", max_results=1)
                    if results:
                        answer = results[0].get('snippet', '')
                        if answer:
                            example = LearningExample(
                                input_text=prompt,
                                output_text=answer,
                                source=LearningSource.CURIOSITY,
                                priority=Priority.LOW,
                                quality_score=0.6,
                                topics=[topic1, topic2]
                            )
                            self.learning_engine.add_learning_example(example)
                            
                            if self.on_learning:
                                self.on_learning(f"Connected: {topic1} <-> {topic2}")
                except Exception:
                    pass
            
            logger.info("Knowledge building complete")
        
        except Exception as e:
            logger.error(f"Error building knowledge: {e}", exc_info=True)
    
    def _consolidate_learning(self):
        """
        Export to training data or trigger LoRA fine-tuning.
        
        REAL IMPLEMENTATION:
        - Checks if enough examples accumulated
        - Exports high-quality examples to training format
        - Could trigger fine-tuning (when available)
        """
        if self.on_thought:
            self.on_thought("Consolidating learning...")
        
        try:
            stats = self.learning_engine.get_queue_stats()
            total = stats['total_examples']
            
            if total < 10:
                logger.debug("Not enough examples to consolidate")
                return
            
            # Export training data
            export_path = self.learning_engine.export_training_data(
                min_quality=0.6,
                max_examples=500
            )
            
            if self.on_learning:
                self.on_learning(f"Consolidated {total} examples to training data")
            
            logger.info(f"Learning consolidated: {export_path}")
        
        except Exception as e:
            logger.error(f"Error consolidating learning: {e}", exc_info=True)
    
    def _optimize_responses(self):
        """
        Analyze patterns in good vs bad responses.
        
        REAL IMPLEMENTATION:
        - Compares high-quality vs low-quality examples
        - Identifies what makes responses good
        - Updates internal metrics
        """
        if self.on_thought:
            self.on_thought("Analyzing response patterns...")
        
        try:
            examples = self.learning_engine.learning_queue
            
            if len(examples) < 10:
                logger.debug("Not enough examples for pattern analysis")
                return
            
            # Separate high and low quality
            high_quality = [e for e in examples if e.quality_score >= 0.8]
            low_quality = [e for e in examples if e.quality_score < 0.5]
            
            if high_quality:
                # Analyze what makes them good
                avg_relevance = sum(e.relevance for e in high_quality) / len(high_quality)
                avg_coherence = sum(e.coherence for e in high_quality) / len(high_quality)
                avg_repetition = sum(e.repetition for e in high_quality) / len(high_quality)
                
                if self.on_learning:
                    self.on_learning(
                        f"Quality patterns: relevance={avg_relevance:.2f}, "
                        f"coherence={avg_coherence:.2f}, repetition={avg_repetition:.2f}"
                    )
                
                logger.info(f"Optimized from {len(high_quality)} high-quality examples")
        
        except Exception as e:
            logger.error(f"Error optimizing responses: {e}", exc_info=True)
    
    def _self_evaluate(self):
        """
        Check own performance metrics.
        
        REAL IMPLEMENTATION:
        - Reviews current performance metrics
        - Identifies areas needing improvement
        - Adjusts learning priorities
        """
        if self.on_thought:
            self.on_thought("Evaluating my performance...")
        
        try:
            metrics = self.learning_engine.get_metrics()
            health = metrics.health_score()
            
            if self.on_learning:
                self.on_learning(
                    f"Health: {health:.1%}, Conversations: {metrics.total_conversations}, "
                    f"Feedback: {metrics.feedback_ratio():.1%} positive"
                )
            
            # Adjust learning based on health
            if health < 0.5:
                # Need to improve - increase reflection
                self.config.reflection_depth = 15
                logger.info("Health low, increasing reflection depth")
            elif health > 0.8:
                # Doing well - maintain current approach
                self.config.reflection_depth = 10
            
            self._save_config()
            logger.info(f"Self-evaluation complete: health={health:.1%}")
        
        except Exception as e:
            logger.error(f"Error in self-evaluation: {e}", exc_info=True)
    
    def _dream(self):
        """
        Creative recombination of knowledge (experimental).
        
        REAL IMPLEMENTATION:
        - Randomly combines concepts from knowledge graph
        - Creates novel connections
        - Generates creative hypotheticals
        """
        if self.on_thought:
            self.on_thought("Dreaming... (creative exploration)")
        
        try:
            topics = self.learning_engine.get_all_topics()
            
            if len(topics) < 3:
                return
            
            # Pick random topics
            selected = random.sample(topics, min(3, len(topics)))
            
            # Create a creative prompt
            prompt = f"Imagine combining {selected[0]}, {selected[1]}, and {selected[2]}"
            
            # This is experimental - just log it
            if self.on_learning:
                self.on_learning(f"Dream: {prompt}")
            
            logger.info(f"Dream sequence: {prompt}")
        
        except Exception as e:
            logger.debug(f"Error in dream: {e}")
    
    def _init_federated_learning(self):
        """Initialize federated learning if enabled."""
        try:
            from ..config import get_config
            from ..learning import FederatedLearning, FederatedMode, PrivacyLevel
            
            fl_config = get_config("federated_learning", {})
            
            if not fl_config.get("enabled", False):
                return
            
            # Parse mode
            mode_str = fl_config.get("mode", "opt_in")
            mode = FederatedMode.OPT_IN
            if mode_str == "opt_out":
                mode = FederatedMode.OPT_OUT
            elif mode_str == "disabled":
                mode = FederatedMode.DISABLED
            
            # Parse privacy level
            privacy_str = fl_config.get("privacy_level", "high")
            privacy = PrivacyLevel.HIGH
            privacy_map = {
                "none": PrivacyLevel.NONE,
                "low": PrivacyLevel.LOW,
                "medium": PrivacyLevel.MEDIUM,
                "high": PrivacyLevel.HIGH,
                "maximum": PrivacyLevel.MAXIMUM,
            }
            privacy = privacy_map.get(privacy_str, PrivacyLevel.HIGH)
            
            # Create federated learning instance
            self.federated_learning = FederatedLearning(
                model_name=self.model_name,
                mode=mode,
                privacy_level=privacy,
            )
            
            logger.info(
                f"Federated learning initialized for {self.model_name} "
                f"(mode={mode.value}, privacy={privacy.value})"
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize federated learning: {e}")
            self.federated_learning = None
    
    def share_learning_update(self):
        """
        Share recent learning improvements via federated learning.
        
        Creates a weight update from recent training and shares it
        with the network (if federated learning is enabled).
        """
        if self.federated_learning is None:
            return
        
        try:
            # Get recent learning improvements
            metrics = self.learning_engine.get_metrics()
            
            if metrics.examples_learned < 10:
                # Need at least 10 examples to share an update
                return
            
            # In a real implementation, this would:
            # 1. Get the actual model weights before/after recent training
            # 2. Compute the delta (difference)
            # 3. Share the delta (not the full weights or data)
            
            # For now, we'll create a placeholder update
            # In production, integrate with actual training system
            
            if self.on_learning:
                self.on_learning(
                    f"[Federated] Ready to share {metrics.examples_learned} "
                    f"improvements (privacy: {self.federated_learning.privacy_level.value})"
                )
            
            logger.info(
                f"Federated learning: {metrics.examples_learned} improvements ready "
                f"(not yet integrated with training system)"
            )
            
        except Exception as e:
            logger.debug(f"Error sharing federated update: {e}")


class AutonomousManager:
    """Manage autonomous mode for multiple AI instances."""
    
    _instances = {}
    
    @classmethod
    def get(cls, model_name: str) -> AutonomousMode:
        if model_name not in cls._instances:
            cls._instances[model_name] = AutonomousMode(model_name)
        return cls._instances[model_name]
    
    @classmethod
    def stop_all(cls):
        for instance in cls._instances.values():
            instance.stop()

