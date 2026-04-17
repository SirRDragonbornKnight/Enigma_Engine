"""Elastic Weight Consolidation (EWC) for continual learning.

Implements Kirkpatrick et al. (2017) — prevents catastrophic forgetting
by penalizing changes to parameters that were important for previous
tasks, as measured by the diagonal of the Fisher information matrix.

Usage:
    # After training on Task A:
    ewc = EWC(model, task_a_dataloader, device)

    # During training on Task B, add to loss:
    loss = ce_loss + ewc.penalty(model)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EWC:
    """Elastic Weight Consolidation penalty.

    Stores a snapshot of model parameters and their Fisher information
    after training on a task.  The penalty for deviating from those
    parameters is:

        penalty = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2

    where F_i is the diagonal Fisher for parameter i, theta*_i is the
    snapshot value, and lambda controls the strength.

    Parameters
    ----------
    model : nn.Module
        Model whose current weights define the "anchor" point.
    data_source : list[torch.Tensor] | Callable[[], torch.Tensor]
        Either a list of input tensors (token ID batches) or a callable
        that yields batches.  Used to estimate Fisher information.
    device : str
        Device for computation.
    n_samples : int
        Number of batches to use for Fisher estimation.  More samples
        = more accurate Fisher, but slower.  50-200 is usually enough.
    lam : float
        Penalty strength (lambda).  Higher = stronger memory of old
        task.  Typical values: 100-10000.
    """

    def __init__(
        self,
        model: nn.Module,
        data_source: "list[torch.Tensor] | Callable[[], torch.Tensor]",
        device: str = "cpu",
        n_samples: int = 100,
        lam: float = 1000.0,
    ):
        self.lam = lam
        self.device = device

        # Snapshot current parameters (the anchor point)
        self._params: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._params[name] = param.data.clone().detach()

        # Estimate diagonal Fisher information
        self._fisher: dict[str, torch.Tensor] = self._compute_fisher(
            model, data_source, n_samples)

        n_params = sum(f.numel() for f in self._fisher.values())
        logger.info(
            "EWC initialized: %d parameters, lambda=%.0f, "
            "%d Fisher samples",
            n_params, lam, n_samples)

    def _compute_fisher(
        self,
        model: nn.Module,
        data_source: Any,
        n_samples: int,
    ) -> dict[str, torch.Tensor]:
        """Estimate diagonal Fisher information matrix.

        Uses the empirical Fisher: average of squared gradients of
        the log-likelihood over data samples.
        """
        fisher: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        model.eval()
        count = 0

        # Normalize data_source to an iterable
        if callable(data_source) and not isinstance(data_source, list):
            batches = (data_source() for _ in range(n_samples))
        else:
            batches = iter(data_source)

        for batch in batches:
            if count >= n_samples:
                break

            model.zero_grad()
            batch = batch.to(self.device)

            # Forward pass — compute log-likelihood (CE loss)
            output = model(batch)
            if isinstance(output, tuple):
                output = output[0]

            # Shift for next-token prediction
            logits = output[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(
                logits.view(-1, logits.size(-1)), dim=-1)
            nll = torch.nn.functional.nll_loss(
                log_probs, targets.view(-1),
                reduction="mean")

            nll.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            count += 1

        # Average over samples
        if count > 0:
            for name in fisher:
                fisher[name] /= count

        model.zero_grad()
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for the current model parameters.

        Returns a scalar tensor with gradient attached, ready to be
        added to the task loss:

            total_loss = task_loss + ewc.penalty(model)

        Returns
        -------
        torch.Tensor
            Scalar EWC penalty (lambda/2 * sum F_i * (theta_i - theta*_i)^2)
        """
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, param in model.named_parameters():
            if name in self._fisher and name in self._params:
                fisher_val = self._fisher[name]
                anchor = self._params[name]
                diff = param - anchor
                loss = loss + (fisher_val * diff.pow(2)).sum()

        return (self.lam / 2.0) * loss

    def save(self, path: str | Path) -> None:
        """Save EWC state (Fisher + anchor params) to disk."""
        from .safe_save import atomic_torch_save
        atomic_torch_save({
            "fisher": self._fisher,
            "params": self._params,
            "lam": self.lam,
        }, str(path))
        logger.info("EWC state saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "EWC":
        """Load EWC state from disk (no model or data needed)."""
        from .model_registry import safe_load_weights
        state = safe_load_weights(str(path), map_location=device)

        obj = cls.__new__(cls)
        obj.device = device
        obj.lam = state["lam"]
        obj._fisher = {
            k: v.to(device) for k, v in state["fisher"].items()}
        obj._params = {
            k: v.to(device) for k, v in state["params"].items()}

        n_params = sum(f.numel() for f in obj._fisher.values())
        logger.info(
            "EWC loaded from %s: %d parameters, lambda=%.0f",
            path, n_params, obj.lam)
        return obj
