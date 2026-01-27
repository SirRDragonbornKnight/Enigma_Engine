"""
Training Coordinator for Federated Learning

Coordinates training rounds across multiple devices:
- Schedule training rounds
- Collect updates from devices
- Trigger aggregation
- Distribute global updates
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

from .aggregation import AggregationMethod, SecureAggregator
from .federated import WeightUpdate

logger = logging.getLogger(__name__)


class RoundStatus(Enum):
    """Status of a training round."""
    PENDING = "pending"        # Not started yet
    ACTIVE = "active"          # Currently training
    AGGREGATING = "aggregating"  # Collecting and aggregating updates
    COMPLETED = "completed"    # Finished
    FAILED = "failed"          # Failed to complete


@dataclass
class TrainingRound:
    """Information about a training round."""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RoundStatus = RoundStatus.PENDING
    updates: List[WeightUpdate] = None
    global_update: Optional[WeightUpdate] = None
    
    def __post_init__(self):
        if self.updates is None:
            self.updates = []


class TrainingCoordinator:
    """
    Coordinate training rounds across devices.
    
    Responsibilities:
    1. Schedule training rounds
    2. Collect updates from devices
    3. Trigger aggregation when enough updates received
    4. Distribute global updates back to devices
    """
    
    def __init__(
        self,
        min_devices: int = 2,
        round_duration: int = 300,  # 5 minutes
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED
    ):
        """
        Initialize training coordinator.
        
        Args:
            min_devices: Minimum number of devices required for aggregation
            round_duration: Duration of each training round in seconds
            aggregation_method: Method for aggregating updates
        """
        self.min_devices = min_devices
        self.round_duration = round_duration
        self.aggregator = SecureAggregator(method=aggregation_method)
        
        # Track training rounds
        self.current_round: Optional[TrainingRound] = None
        self.completed_rounds: List[TrainingRound] = []
        
        # Callbacks
        self._on_round_start: List[Callable] = []
        self._on_round_complete: List[Callable] = []
        
        # Coordination thread
        self._coordinator_thread = None
        self._running = False
        
        logger.info(
            f"Initialized TrainingCoordinator: "
            f"min_devices={min_devices}, "
            f"round_duration={round_duration}s"
        )
    
    def on_round_start(self, callback: Callable[[int], None]) -> None:
        """
        Register callback for round start.
        
        Args:
            callback: Function to call when round starts (takes round_id)
        """
        self._on_round_start.append(callback)
    
    def on_round_complete(self, callback: Callable[[TrainingRound], None]) -> None:
        """
        Register callback for round completion.
        
        Args:
            callback: Function to call when round completes (takes TrainingRound)
        """
        self._on_round_complete.append(callback)
    
    def start_coordination(self) -> None:
        """Start the coordination thread."""
        if self._running:
            logger.warning("Coordinator already running")
            return
        
        self._running = True
        self._coordinator_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self._coordinator_thread.start()
        
        logger.info("Started training coordination")
    
    def stop_coordination(self) -> None:
        """Stop the coordination thread."""
        self._running = False
        if self._coordinator_thread:
            self._coordinator_thread.join(timeout=5.0)
        
        logger.info("Stopped training coordination")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self._running:
            try:
                # Check if we need to start a new round
                if self.current_round is None:
                    self._start_new_round()
                
                # Check if current round has timed out
                elif self._is_round_expired():
                    self._complete_round()
                
                # Sleep briefly
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(5.0)
    
    def _start_new_round(self) -> None:
        """Start a new training round."""
        round_id = len(self.completed_rounds) + 1
        
        self.current_round = TrainingRound(
            round_id=round_id,
            start_time=datetime.now(),
            status=RoundStatus.ACTIVE
        )
        
        logger.info(f"Started training round {round_id}")
        
        # Notify callbacks
        for callback in self._on_round_start:
            try:
                callback(round_id)
            except Exception as e:
                logger.error(f"Error in round start callback: {e}")
    
    def _is_round_expired(self) -> bool:
        """Check if current round has expired."""
        if not self.current_round:
            return False
        
        elapsed = (datetime.now() - self.current_round.start_time).total_seconds()
        return elapsed >= self.round_duration
    
    def _complete_round(self) -> None:
        """Complete the current training round."""
        if not self.current_round:
            return
        
        self.current_round.status = RoundStatus.AGGREGATING
        
        # Check if we have enough updates
        if len(self.current_round.updates) < self.min_devices:
            logger.warning(
                f"Round {self.current_round.round_id} has insufficient updates: "
                f"{len(self.current_round.updates)} < {self.min_devices}"
            )
            self.current_round.status = RoundStatus.FAILED
        else:
            # Aggregate updates
            try:
                # Validate updates first
                valid_updates = self.aggregator.validate_updates(
                    self.current_round.updates
                )
                
                if len(valid_updates) >= self.min_devices:
                    # Aggregate
                    global_update = self.aggregator.aggregate_updates(valid_updates)
                    self.current_round.global_update = global_update
                    self.current_round.status = RoundStatus.COMPLETED
                    
                    logger.info(
                        f"Completed round {self.current_round.round_id}: "
                        f"aggregated {len(valid_updates)} updates"
                    )
                else:
                    logger.warning(
                        f"Round {self.current_round.round_id} has insufficient "
                        f"valid updates: {len(valid_updates)} < {self.min_devices}"
                    )
                    self.current_round.status = RoundStatus.FAILED
                    
            except Exception as e:
                logger.error(f"Failed to aggregate round: {e}")
                self.current_round.status = RoundStatus.FAILED
        
        # Finalize round
        self.current_round.end_time = datetime.now()
        
        # Notify callbacks
        for callback in self._on_round_complete:
            try:
                callback(self.current_round)
            except Exception as e:
                logger.error(f"Error in round complete callback: {e}")
        
        # Move to completed rounds
        self.completed_rounds.append(self.current_round)
        self.current_round = None
    
    def submit_update(self, update: WeightUpdate) -> bool:
        """
        Submit a weight update for the current round.
        
        Args:
            update: Weight update from a device
            
        Returns:
            True if accepted
        """
        if not self.current_round:
            logger.warning("No active round - cannot submit update")
            return False
        
        if self.current_round.status != RoundStatus.ACTIVE:
            logger.warning(
                f"Round {self.current_round.round_id} is not active - "
                f"cannot submit update"
            )
            return False
        
        # Add to current round
        self.current_round.updates.append(update)
        
        logger.debug(
            f"Received update {update.update_id[:8]}... for round "
            f"{self.current_round.round_id} "
            f"({len(self.current_round.updates)} total)"
        )
        
        # Check if we have enough updates to complete early
        if len(self.current_round.updates) >= self.min_devices * 2:
            logger.info(
                f"Round {self.current_round.round_id} has sufficient updates, "
                f"completing early"
            )
            self._complete_round()
        
        return True
    
    def get_current_round_id(self) -> Optional[int]:
        """Get the current round ID."""
        return self.current_round.round_id if self.current_round else None
    
    def get_round_status(self, round_id: int) -> Optional[TrainingRound]:
        """Get status of a specific round."""
        # Check current round
        if self.current_round and self.current_round.round_id == round_id:
            return self.current_round
        
        # Check completed rounds
        for round_info in self.completed_rounds:
            if round_info.round_id == round_id:
                return round_info
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get coordination statistics."""
        total_rounds = len(self.completed_rounds)
        successful_rounds = sum(
            1 for r in self.completed_rounds 
            if r.status == RoundStatus.COMPLETED
        )
        
        return {
            "total_rounds": total_rounds,
            "successful_rounds": successful_rounds,
            "failed_rounds": total_rounds - successful_rounds,
            "current_round": self.current_round.round_id if self.current_round else None,
            "min_devices": self.min_devices,
            "round_duration": self.round_duration,
        }
