"""
GNN graph attribution for fraud ring detection.

Provides per-transaction graph attribution scores indicating which
graph relationships (shared devices, mule transfers, merchant patterns)
contributed to the fraud ring risk score.

Since real-time GNNExplainer requires the full subgraph (expensive at
inference time), this module uses two approaches:

1. OFFLINE (batch): Full PyG GNNExplainer on the stored graph — used
   during batch re-scoring and model evaluation.

2. ONLINE (per-transaction): Lightweight heuristic attribution from
   pre-computed account risk embeddings — used at /explain request time.
   Runs in <5ms per transaction.

Full graph-level GNNExplainer is deferred to Phase 10 batch monitoring.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def compute_online_attribution(
    account_id: str,
    sender_device_id: str,
    receiver_account_id: str | None,
    sim_swap_detected: bool,
    gnn_score: float,
    enriched: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Lightweight graph attribution for real-time /explain responses.

    Returns a dict of edge_type → attribution_score indicating which
    graph relationship types contributed to the GNN fraud ring score.

    Attribution values are in [0, 1]. Higher = stronger contribution.
    """
    attribution: dict[str, float] = {}

    if gnn_score is None or gnn_score == 0.0:
        return attribution

    # Device sharing signal
    # If device_change_24h > 0, account has used multiple devices — ring indicator
    device_change = float((enriched or {}).get("device_change_24h", 0))
    device_attr = min(1.0, device_change * 0.3) * gnn_score
    attribution["account_uses_device"] = round(device_attr, 4)

    # Peer transfer signal — PayShap/EFT to another account is a ring drain pattern
    if receiver_account_id is not None:
        transfer_attr = gnn_score * 0.8
        attribution["account_transfers_to"] = round(transfer_attr, 4)

    # SIM swap + new device + transfer = classic ring pattern
    if sim_swap_detected and receiver_account_id is not None:
        ring_attr = min(1.0, gnn_score * 1.1)
        attribution["sim_swap_ring_pattern"] = round(ring_attr, 4)

    # Velocity signal — high tx count in 1hr is shared-device ring behaviour
    vel_1h = float((enriched or {}).get("tx_count_1hr", 0))
    if vel_1h > 3:
        vel_attr = min(1.0, (vel_1h / 10.0) * gnn_score)
        attribution["velocity_cluster"] = round(vel_attr, 4)

    return attribution


def run_offline_gnn_explainer(
    graph_path: str,
    account_id: str,
    model: Any,
    device: str = "cpu",
    num_hops: int = 2,
) -> dict[str, float]:
    """
    Run PyG GNNExplainer on the stored graph for a given account.
    Used for batch re-scoring and model evaluation — not called at /explain time.

    Returns edge_mask dict keyed by edge_type.
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer

        graph = torch.load(graph_path, map_location=device, weights_only=False)
        model.eval()

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",
                return_type="probs",
            ),
        )

        # Find node index for this account
        account_ids = graph["account"].account_id if hasattr(graph["account"], "account_id") else []
        if account_id not in account_ids:
            return {}

        node_idx = list(account_ids).index(account_id)
        x_dict = {nt: graph[nt].x for nt in graph.node_types if hasattr(graph[nt], "x")}
        edge_index_dict = {et: graph[et].edge_index for et in graph.edge_types}

        explanation = explainer(x_dict, edge_index_dict, index=node_idx)

        result: dict[str, float] = {}
        for et in graph.edge_types:
            et_str = f"{et[0]}__{et[1]}__{et[2]}"
            if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None:
                result[et_str] = float(explanation.edge_mask.mean())

        return result

    except Exception as e:
        logger.warning(f"Offline GNN explainer failed for {account_id}: {e}")
        return {}
