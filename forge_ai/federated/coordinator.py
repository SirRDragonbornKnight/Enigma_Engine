"""
================================================================================
FEDERATED COORDINATOR - MANAGE TRAINING ROUNDS
================================================================================

Coordinates federated learning rounds across multiple participants.
Usually runs on the most powerful device.

FILE: forge_ai/federated/coordinator.py
TYPE: Training Coordination
MAIN CLASS: FederatedCoordinator

HOW IT WORKS:
    1. Broadcast "start round N"
    2. Wait for participants to train
    3. Collect updates (with timeout)
    4. Aggregate updates
    5. Broadcast improved model
    6. Wait before next round

USAGE:
    coordinator = FederatedCoordinator()
    await coordinator.run_round()
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from .federation import ModelUpdate, FederationInfo
from .aggregation import FederatedAggregator

logger = logging.getLogger(__name__)


class FederatedCoordinator:
    """
    Coordinates federated learning rounds.
    
    Usually runs on most powerful device.
    """
    
    def __init__(
        self,
        min_participants: int = 2,
        round_timeout: int = 300,
        wait_between_rounds: int = 60
    ):
        """
        Initialize coordinator.
        
        Args:
            min_participants: Minimum participants required per round
            round_timeout: Timeout for waiting for updates (seconds)
            wait_between_rounds: Wait time between rounds (seconds)
        """
        self.participants: List[str] = []
        self.current_round = 0
        self.min_participants = min_participants
        self.round_timeout = round_timeout
        self.wait_between_rounds = wait_between_rounds
        
        self.aggregator = FederatedAggregator()
        self.round_history: List[Dict] = []
        
        # Current model weights (coordinator maintains base model)
        self.base_weights: Optional[Dict[str, np.ndarray]] = None
        
        logger.info(
            f"Initialized coordinator: min_participants={min_participants}, "
            f"timeout={round_timeout}s"
        )
    
    def register_participant(self, device_id: str):
        """
        Register a new participant.
        
        Args:
            device_id: Device ID to register
        """
        if device_id not in self.participants:
            self.participants.append(device_id)
            logger.info(f"Registered participant {device_id} ({len(self.participants)} total)")
    
    def unregister_participant(self, device_id: str):
        """
        Unregister a participant.
        
        Args:
            device_id: Device ID to unregister
        """
        if device_id in self.participants:
            self.participants.remove(device_id)
            logger.info(f"Unregistered participant {device_id} ({len(self.participants)} remaining)")
    
    async def run_round(self) -> Dict:
        """
        Execute one federated learning round.
        
        Steps:
        1. Broadcast "start round N"
        2. Wait for participants to train
        3. Collect updates (with timeout)
        4. Aggregate updates
        5. Broadcast improved model
        6. Wait before next round
        
        Returns:
            Round statistics
        """
        round_start = datetime.now()
        logger.info(f"Starting round {self.current_round}")
        
        # Check if we have enough participants
        if len(self.participants) < self.min_participants:
            logger.warning(
                f"Not enough participants ({len(self.participants)} < {self.min_participants}), "
                "skipping round"
            )
            return {
                'round': self.current_round,
                'status': 'skipped',
                'reason': 'insufficient_participants',
            }
        
        # 1. Broadcast start round
        await self._broadcast_start_round()
        
        # 2. Collect updates
        updates = await self.collect_updates(
            timeout=self.round_timeout,
            min_updates=self.min_participants
        )
        
        # 3. Check if we got enough updates
        if len(updates) < self.min_participants:
            logger.warning(
                f"Insufficient updates received ({len(updates)} < {self.min_participants}), "
                "skipping aggregation"
            )
            return {
                'round': self.current_round,
                'status': 'failed',
                'reason': 'insufficient_updates',
                'received': len(updates),
                'required': self.min_participants,
            }
        
        # 4. Aggregate updates
        aggregated_deltas = self.aggregator.aggregate_updates(updates)
        
        # 5. Apply updates to base model (if we have one)
        if self.base_weights:
            self.base_weights = self.aggregator.apply_updates(
                self.base_weights,
                aggregated_deltas
            )
        else:
            # First round - aggregated deltas become base weights
            self.base_weights = aggregated_deltas
        
        # 6. Broadcast improved model
        await self._broadcast_model(self.base_weights)
        
        # Record round stats
        round_end = datetime.now()
        round_stats = {
            'round': self.current_round,
            'status': 'completed',
            'participants': len(updates),
            'total_samples': sum(u.num_samples for u in updates),
            'avg_loss': sum(u.loss for u in updates) / len(updates),
            'duration': (round_end - round_start).total_seconds(),
            'timestamp': round_end.isoformat(),
        }
        
        self.round_history.append(round_stats)
        logger.info(
            f"Round {self.current_round} completed: {len(updates)} updates, "
            f"avg_loss={round_stats['avg_loss']:.4f}"
        )
        
        self.current_round += 1
        
        return round_stats
    
    async def _broadcast_start_round(self):
        """
        Broadcast "start round N" message to all participants.
        
        In a real implementation, this would send over network.
        """
        message = {
            "type": "start_round",
            "round": self.current_round,
            "timeout": self.round_timeout,
        }
        
        logger.debug(f"Broadcasting start round {self.current_round}")
        # In real implementation: send to all participants via network
        await asyncio.sleep(0.1)  # Simulate network delay
    
    async def collect_updates(
        self,
        timeout: int,
        min_updates: int
    ) -> List[ModelUpdate]:
        """
        Wait for updates from participants.
        
        Args:
            timeout: Maximum time to wait (seconds)
            min_updates: Minimum number of updates required
        
        Returns:
            List of model updates received
        """
        logger.info(f"Collecting updates (timeout={timeout}s, min={min_updates})")
        
        updates = []
        start_time = asyncio.get_event_loop().time()
        
        # In real implementation, this would listen for incoming updates
        # For now, just wait for timeout
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed >= timeout:
                logger.info(f"Timeout reached, collected {len(updates)} updates")
                break
            
            if len(updates) >= len(self.participants):
                logger.info("Received updates from all participants")
                break
            
            # Check for new updates (placeholder)
            # In real implementation: check network queue
            await asyncio.sleep(1)
        
        return updates
    
    async def _broadcast_model(self, model_weights: Dict[str, np.ndarray]):
        """
        Broadcast improved model to all participants.
        
        Args:
            model_weights: Updated model weights
        """
        logger.info(f"Broadcasting improved model ({len(model_weights)} layers)")
        
        # In real implementation: send to all participants via network
        await asyncio.sleep(0.1)  # Simulate network delay
    
    def get_stats(self) -> Dict:
        """
        Get coordinator statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            'current_round': self.current_round,
            'participants': len(self.participants),
            'total_rounds': len(self.round_history),
            'successful_rounds': len([r for r in self.round_history if r['status'] == 'completed']),
            'aggregator_stats': self.aggregator.get_stats(),
        }
    
    def get_round_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get round history.
        
        Args:
            last_n: Number of recent rounds to return (None = all)
        
        Returns:
            List of round statistics
        """
        if last_n:
            return self.round_history[-last_n:]
        return self.round_history.copy()
    
    async def run_continuous(
        self,
        num_rounds: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None
    ):
        """
        Run continuous training rounds.
        
        Args:
            num_rounds: Number of rounds to run (None = infinite)
            stop_event: Event to signal stopping
        """
        rounds_completed = 0
        
        while True:
            # Check stop conditions
            if stop_event and stop_event.is_set():
                logger.info("Stop event received, ending training")
                break
            
            if num_rounds and rounds_completed >= num_rounds:
                logger.info(f"Completed {num_rounds} rounds")
                break
            
            # Run one round
            await self.run_round()
            rounds_completed += 1
            
            # Wait before next round
            logger.info(f"Waiting {self.wait_between_rounds}s before next round")
            await asyncio.sleep(self.wait_between_rounds)
