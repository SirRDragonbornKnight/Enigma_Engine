"""
================================================================================
FEDERATED COORDINATOR - MANAGE TRAINING ROUNDS
================================================================================

Coordinates federated learning rounds across multiple participants.
Usually runs on the most powerful device.

FILE: enigma_engine/federated/coordinator.py
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
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .aggregation import FederatedAggregator
from .federation import FederationInfo, ModelUpdate

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
        self.participants: list[str] = []
        self.current_round = 0
        self.min_participants = min_participants
        self.round_timeout = round_timeout
        self.wait_between_rounds = wait_between_rounds
        
        self.aggregator = FederatedAggregator()
        self.round_history: list[dict] = []
        
        # Current model weights (coordinator maintains base model)
        self.base_weights: Optional[dict[str, np.ndarray]] = None
        
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
    
    async def run_round(self) -> dict:
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
        # Trim history to prevent unbounded growth
        if len(self.round_history) > self._max_round_history:
            self.round_history = self.round_history[-self._max_round_history:]
        logger.info(
            f"Round {self.current_round} completed: {len(updates)} updates, "
            f"avg_loss={round_stats['avg_loss']:.4f}"
        )
        
        self.current_round += 1
        
        return round_stats
    
    async def _broadcast_start_round(self):
        """
        Broadcast "start round N" message to all participants.
        
        Sends the round start notification over the network to registered participants.
        """
        message = {
            "type": "start_round",
            "round": self.current_round,
            "timeout": self.round_timeout,
        }
        
        logger.debug(f"Broadcasting start round {self.current_round} to {len(self.participants)} participants")
        
        # Send to each participant
        errors = []
        for participant_id in self.participants:
            try:
                await self._send_to_participant(participant_id, message)
            except Exception as e:
                errors.append((participant_id, str(e)))
                logger.warning(f"Failed to notify participant {participant_id}: {e}")
        
        if errors:
            logger.warning(f"Failed to notify {len(errors)} participants")
        
        # Small delay to ensure messages are received
        await asyncio.sleep(0.05)
    
    async def _send_to_participant(self, participant_id: str, message: dict):
        """
        Send a message to a specific participant.
        
        Args:
            participant_id: ID of the participant
            message: Message to send
        """
        # Check if we have network connection info
        if hasattr(self, '_participant_connections') and participant_id in self._participant_connections:
            conn_info = self._participant_connections[participant_id]
            try:
                import json as json_lib
                import urllib.request
                
                address = conn_info.get('address', 'localhost')
                port = conn_info.get('port', 5000)
                url = f"http://{address}:{port}/federated/message"
                
                req = urllib.request.Request(
                    url,
                    data=json_lib.dumps(message).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    logger.debug(f"Sent message to {participant_id}: {response.status}")
                    
            except Exception as e:
                logger.debug(f"HTTP send to {participant_id} failed: {e}, using local queue")
                # Fall back to local queue for same-process participants
                if hasattr(self, '_message_queues') and participant_id in self._message_queues:
                    self._message_queues[participant_id].put(message)
        else:
            # Local mode - use message queue if available
            if hasattr(self, '_message_queues') and participant_id in self._message_queues:
                self._message_queues[participant_id].put(message)
            else:
                logger.debug(f"No connection info for {participant_id}, message queued locally")
    
    async def collect_updates(
        self,
        timeout: int,
        min_updates: int
    ) -> list[ModelUpdate]:
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
        start_time = asyncio.get_running_loop().time()
        
        # In real implementation, this would listen for incoming updates
        # For now, just wait for timeout
        while True:
            elapsed = asyncio.get_running_loop().time() - start_time
            
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
    
    async def _broadcast_model(self, model_weights: dict[str, np.ndarray]):
        """
        Broadcast improved model to all participants.
        
        Args:
            model_weights: Updated model weights
        """
        logger.info(f"Broadcasting improved model ({len(model_weights)} layers)")
        
        # Serialize weights
        serialized = {}
        for name, weights in model_weights.items():
            serialized[name] = weights.tolist() if hasattr(weights, 'tolist') else weights
        
        message = {
            "type": "model_update",
            "round": self.current_round,
            "weights": serialized,
            "num_layers": len(model_weights),
        }
        
        # Check total size
        import sys
        size_estimate = sys.getsizeof(str(serialized))
        if size_estimate > 10_000_000:  # 10MB threshold
            logger.warning(f"Model update is large ({size_estimate/1e6:.1f}MB), consider compression")
        
        # Send to each participant
        success_count = 0
        for participant_id in self.participants:
            try:
                await self._send_to_participant(participant_id, message)
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to send model to {participant_id}: {e}")
        
        logger.info(f"Model broadcast complete: {success_count}/{len(self.participants)} participants updated")
    
    def register_participant_connection(self, participant_id: str, address: str, port: int):
        """
        Register connection info for a participant.
        
        Args:
            participant_id: Participant's device ID
            address: IP address or hostname
            port: Port number
        """
        if not hasattr(self, '_participant_connections'):
            self._participant_connections = {}
        
        self._participant_connections[participant_id] = {
            'address': address,
            'port': port,
        }
        
        # Also register the participant if not already
        if participant_id not in self.participants:
            self.register_participant(participant_id)
        
        logger.info(f"Registered connection for {participant_id} at {address}:{port}")
    
    def get_stats(self) -> dict:
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
    
    def get_round_history(self, last_n: Optional[int] = None) -> list[dict]:
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
