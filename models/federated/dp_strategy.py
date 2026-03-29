"""
Differentially private FedAvg strategy.

Extends Flower's FedAvg with central Gaussian DP noise on the aggregated update.
Privacy budget (epsilon) is tracked per round using the RDP accountant from
dp_accounting.

Central DP model:
  - Each client clips its update to clip_norm before sending (done in client.py)
  - Server adds Gaussian noise N(0, sigma^2 * I) to the mean aggregate
  - sigma = noise_multiplier * clip_norm / num_clients
  - Privacy budget tracked via RDP accountant
"""

from __future__ import annotations

import logging

import numpy as np
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class DPFedAvg(FedAvg):
    """
    FedAvg + central Gaussian differential privacy.

    Args:
        noise_multiplier: Gaussian noise multiplier (sigma = noise_multiplier * clip_norm / n)
        clip_norm:        L2 norm bound on each client's update (clipping done client-side)
        num_clients:      Total number of participating clients
        delta:            Target delta for (epsilon, delta)-DP
        **fedavg_kwargs:  Passed through to FedAvg
    """

    def __init__(
        self,
        *,
        noise_multiplier: float = 1.0,
        clip_norm: float = 1.0,
        num_clients: int = 5,
        delta: float = 1e-5,
        **fedavg_kwargs,
    ) -> None:
        super().__init__(**fedavg_kwargs)
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.num_clients = num_clients
        self.delta = delta
        self._round = 0
        self._epsilon = 0.0

    @property
    def epsilon(self) -> float:
        """Current privacy budget spent."""
        return self._epsilon

    def _compute_epsilon(self, num_rounds: int) -> float:
        """Compute epsilon after num_rounds using RDP accountant."""
        accountant = rdp_privacy_accountant.RdpAccountant()
        # All clients participate each round (q=1.0)
        event = dp_event.SelfComposedDpEvent(
            dp_event.GaussianDpEvent(noise_multiplier=self.noise_multiplier),
            count=num_rounds,
        )
        accountant.compose(event)
        epsilon, _ = accountant.get_epsilon_and_optimal_order(self.delta)
        return float(epsilon)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate with FedAvg then add Gaussian noise."""
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is None:
            return None, metrics

        # Add Gaussian noise to mean aggregate
        # sigma for central DP on mean = noise_multiplier * clip_norm / num_clients
        sigma = self.noise_multiplier * self.clip_norm / self.num_clients
        ndarrays = parameters_to_ndarrays(aggregated_params)
        noisy = [
            arr + np.random.normal(0, sigma, size=arr.shape).astype(arr.dtype) for arr in ndarrays
        ]

        self._round = server_round
        self._epsilon = self._compute_epsilon(server_round)
        logger.info(f"Round {server_round}: epsilon={self._epsilon:.4f} (delta={self.delta})")

        return ndarrays_to_parameters(noisy), {**metrics, "epsilon": self._epsilon}
