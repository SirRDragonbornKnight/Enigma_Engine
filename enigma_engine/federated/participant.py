"""
================================================================================
FEDERATED PARTICIPANT - CONTRIBUTE TO FEDERATION
================================================================================

Federated learning participant that trains locally and shares updates.
Runs on every device that wants to contribute.

FILE: enigma_engine/federated/participant.py
TYPE: Participant Client
MAIN CLASS: FederatedParticipant

HOW IT WORKS:
    1. Listen for training round announcements
    2. Train on local data
    3. Send update to coordinator
    4. Wait for improved model
    5. Apply improved model

USAGE:
    participant = FederatedParticipant()
    await participant.participate_in_round(round_number)
"""

import asyncio
import logging
from typing import Optional

import numpy as np

from ..config import CONFIG
from .compression import UpdateCompressor
from .federation import ModelUpdate
from .privacy import DifferentialPrivacy

logger = logging.getLogger(__name__)


class FederatedParticipant:
    """
    Federated learning participant.
    
    Runs on every device that wants to contribute.
    """
    
    def __init__(self):
        """Initialize participant."""
        self.coordinator_url: Optional[str] = None
        self.current_round = 0
        self.device_id = self._get_device_id()
        
        # Get config
        fed_config = CONFIG.get("federated_learning", {})
        
        # Privacy protection
        privacy_config = fed_config.get("privacy", {})
        if privacy_config.get("differential_privacy", True):
            epsilon = privacy_config.get("epsilon", 1.0)
            self.privacy = DifferentialPrivacy(epsilon=epsilon)
        else:
            self.privacy = None
        
        # Compression
        compression_config = fed_config.get("compression", {})
        if compression_config.get("enabled", True):
            self.compressor = UpdateCompressor(
                quantization_bits=compression_config.get("quantization_bits", 8),
                sparsity=compression_config.get("sparsity", 0.1),
            )
        else:
            self.compressor = None
        
        # Resource limits
        self.max_training_time = fed_config.get("resources", {}).get("max_training_time", 300)
        self.min_samples = fed_config.get("resources", {}).get("min_samples", 10)
        
        logger.info(f"Initialized participant {self.device_id}")
    
    def _get_device_id(self) -> str:
        """Get device ID."""
        from pathlib import Path
        device_id_file = Path(CONFIG.get("info_dir", "information")) / "device_id.txt"
        device_id_file.parent.mkdir(parents=True, exist_ok=True)
        
        if device_id_file.exists():
            return device_id_file.read_text().strip()
        else:
            import uuid
            device_id = str(uuid.uuid4())
            device_id_file.write_text(device_id)
            return device_id
    
    async def listen_for_rounds(self, coordinator_url: str):
        """
        Listen for training round announcements.
        
        When round starts:
        1. Train on local data
        2. Send update to coordinator
        3. Wait for improved model
        4. Apply improved model
        
        Args:
            coordinator_url: URL of coordinator
        """
        self.coordinator_url = coordinator_url
        logger.info(f"Listening for rounds from {coordinator_url}")
        
        # In real implementation, listen for network messages
        # For now, just placeholder
        while True:
            await asyncio.sleep(10)
    
    async def participate_in_round(
        self,
        round_number: int,
        model = None,
        dataset = None
    ) -> dict:
        """
        Participate in a training round.
        
        Args:
            round_number: Round number
            model: Model to train (if None, use loaded model)
            dataset: Local dataset (if None, use saved data)
        
        Returns:
            Statistics about participation
        """
        logger.info(f"Participating in round {round_number}")
        
        # Check if we have minimum samples
        dataset_size = self._get_dataset_size(dataset)
        if dataset and dataset_size < self.min_samples:
            logger.warning(f"Insufficient samples ({dataset_size} < {self.min_samples}), skipping")
            return {
                'status': 'skipped',
                'reason': 'insufficient_samples',
            }
        
        try:
            # 1. Train locally
            update = await self._train_on_local_data(model, dataset, round_number)
            
            if not update:
                return {
                    'status': 'failed',
                    'reason': 'training_failed',
                }
            
            # 2. Add privacy protection
            if self.privacy:
                update = self.privacy.add_noise(update)
                logger.debug("Added differential privacy protection")
            
            # 3. Compress for transfer
            if self.compressor:
                compressed = self.compressor.compress(update)
                logger.debug(f"Compressed update: {compressed.compression_ratio*100:.1f}% of original")
            else:
                compressed = update
            
            # 4. Send to coordinator
            await self._send_update(compressed)
            
            # 5. Wait for improved model
            improved_model = await self._receive_model()
            
            # 6. Apply improvements
            if improved_model:
                self._apply_model_update(improved_model)
            
            self.current_round = round_number
            
            return {
                'status': 'completed',
                'round': round_number,
                'num_samples': update.num_samples,
                'loss': update.loss,
            }
            
        except Exception as e:
            logger.error(f"Error participating in round: {e}")
            return {
                'status': 'error',
                'error': str(e),
            }
    
    async def _train_on_local_data(
        self,
        model,
        dataset,
        round_number: int
    ) -> Optional[ModelUpdate]:
        """
        Train on local data using Enigma AI Engine training infrastructure.
        
        Args:
            model: Model to train
            dataset: Local dataset
            round_number: Current round number
        
        Returns:
            Model update or None if training failed
        """
        import torch
        
        logger.info("Training on local data...")
        
        if not model or not dataset:
            logger.warning("No model or dataset provided")
            return None
        
        # Store initial weights
        initial_weights = {}
        try:
            for name, param in model.named_parameters():
                initial_weights[name] = param.detach().cpu().numpy().copy()
        except Exception as e:
            logger.error(f"Error storing initial weights: {e}")
            return None
        
        # Train using Enigma AI Engine training infrastructure
        loss = 0.0
        try:
            from enigma_engine.core.training import TrainingConfig
            
            config = TrainingConfig(
                epochs=1,  # Single epoch per federated round
                learning_rate=getattr(self, 'learning_rate', 1e-4),
                batch_size=getattr(self, 'batch_size', 4),
                save_checkpoints=False,
            )
            
            # Run training in executor to not block async loop
            import concurrent.futures
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(
                    executor,
                    self._run_training_sync,
                    model, dataset, config
                )
                try:
                    loss = await asyncio.wait_for(future, timeout=self.max_training_time)
                except asyncio.TimeoutError:
                    logger.warning(f"Training exceeded max time ({self.max_training_time}s)")
                    
        except ImportError:
            # Fallback: basic training loop
            logger.debug("Using fallback training")
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(self, 'learning_rate', 1e-4))
            
            dataset_size = self._get_dataset_size(dataset)
            batch_size = getattr(self, 'batch_size', 4)
            
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, min(dataset_size, 100), batch_size):  # Limit iterations
                try:
                    batch = dataset[i:i + batch_size] if hasattr(dataset, '__getitem__') else next(iter(dataset))
                    
                    if isinstance(batch, dict):
                        input_ids = batch.get('input_ids')
                    elif isinstance(batch, (list, tuple)):
                        input_ids = batch[0]
                    else:
                        continue
                    
                    if input_ids is None:
                        continue
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids)
                    
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        batch_loss = outputs.loss
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item()
                        num_batches += 1
                except Exception as e:
                    logger.debug(f"Batch error: {e}")
                    continue
            
            loss = total_loss / max(num_batches, 1)
        
        # Calculate weight deltas
        weight_deltas = {}
        try:
            for name, param in model.named_parameters():
                current_weight = param.detach().cpu().numpy()
                weight_deltas[name] = current_weight - initial_weights[name]
        except Exception as e:
            logger.error(f"Error calculating deltas: {e}")
            return None
        
        # Create update
        dataset_size = self._get_dataset_size(dataset)
        update = ModelUpdate(
            device_id=self.device_id,
            round_number=round_number,
            weight_deltas=weight_deltas,
            num_samples=dataset_size,
            loss=loss,
        )
        
        logger.info(f"Training completed: {len(weight_deltas)} layers, loss={loss:.4f}")
        return update
    
    def _run_training_sync(self, model, dataset, config):
        """Run training synchronously (called from executor)."""
        try:
            from enigma_engine.core.training import Trainer
            trainer = Trainer(model, config)
            result = trainer.train(dataset)
            return result.get('final_loss', 0.0) if result else 0.0
        except Exception as e:
            logger.error(f"Training error: {e}")
            return 0.0
    
    async def _send_update(self, update):
        """
        Send update to coordinator.
        
        Args:
            update: Update to send (compressed or uncompressed)
        """
        logger.debug("Sending update to coordinator")
        
        # In real implementation, send over network
        await asyncio.sleep(0.1)  # Simulate network delay
    
    async def _receive_model(self) -> Optional[dict[str, np.ndarray]]:
        """
        Wait for improved model from coordinator.
        
        Returns:
            Model weights or None if timeout
        """
        logger.debug("Waiting for improved model...")
        
        # In real implementation, listen for network message
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Placeholder - return None
        return None
    
    def _apply_model_update(self, model_weights: dict[str, np.ndarray]):
        """
        Apply improved model to local model.
        
        Args:
            model_weights: Updated model weights
        """
        logger.info(f"Applying model update ({len(model_weights)} layers)")
        
        # In real implementation, apply to local model
        # For now, just log
    
    def _get_dataset_size(self, dataset) -> int:
        """
        Get size of dataset safely.
        
        Args:
            dataset: Dataset object
        
        Returns:
            Size of dataset, or 0 if unknown
        """
        if dataset is None:
            return 0
        
        # Try __len__ first
        if hasattr(dataset, '__len__'):
            try:
                return len(dataset)
            except Exception:
                pass
        
        # Try size attribute
        if hasattr(dataset, 'size'):
            try:
                return dataset.size
            except Exception:
                pass
        
        # Unknown size
        return 0
    
    def get_stats(self) -> dict:
        """
        Get participant statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            'device_id': self.device_id,
            'current_round': self.current_round,
            'coordinator_url': self.coordinator_url,
            'privacy_enabled': self.privacy is not None,
            'compression_enabled': self.compressor is not None,
        }
