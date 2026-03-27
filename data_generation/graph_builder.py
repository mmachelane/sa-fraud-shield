"""
Build a PyG HeteroData graph from the raw transactions + accounts parquet files.

Node types:
    account       — bank account (labelled: is_fraud_ring_member)
    device        — device fingerprint
    merchant      — merchant ID

Edge types:
    (account, transacts_with, merchant)   — account made a transaction at merchant
    (account, uses_device, device)        — account used this device
    (device,  shared_by,   account)       — reverse of uses_device (for message passing)
    (account, payshap_transfer, account)  — PayShap peer transfer
    (account, eft_transfer, account)      — EFT transfer

Usage:
    python -m data_generation.graph_builder \
        --transactions-path data/raw/transactions.parquet \
        --output-path data/graphs/hetero_graph.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def build_graph(
    transactions_path: Path | str,
    accounts_path: Path | str,
    output_path: Path | str,
) -> HeteroData:
    transactions_path = Path(transactions_path)
    accounts_path = Path(accounts_path)
    output_path = Path(output_path)

    logger.info("Loading raw data...")
    tx = pd.read_parquet(transactions_path)
    accounts = pd.read_parquet(accounts_path)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    logger.info(f"Transactions: {len(tx):,}  |  Accounts: {len(accounts):,}")

    # ── Build index maps (string ID → integer index) ──────────────────────────
    account_ids = accounts["account_id"].tolist()
    account_idx = {aid: i for i, aid in enumerate(account_ids)}

    device_ids = tx["sender_device_id"].dropna().unique().tolist()
    device_idx = {did: i for i, did in enumerate(device_ids)}

    merchant_ids = tx["merchant_id"].dropna().unique().tolist()
    merchant_idx = {mid: i for i, mid in enumerate(merchant_ids)}

    n_accounts = len(account_ids)
    n_devices = len(device_ids)
    n_merchants = len(merchant_ids)

    logger.info(
        f"Nodes — accounts: {n_accounts:,}  devices: {n_devices:,}  merchants: {n_merchants:,}"
    )

    # ── Account node features ─────────────────────────────────────────────────
    # Normalise continuous features; keep only numeric, no PII
    acc_features = accounts[["monthly_income_zar", "account_age_days"]].copy()
    acc_features["monthly_income_zar"] = acc_features["monthly_income_zar"].fillna(0).astype(float)
    acc_features["account_age_days"] = acc_features["account_age_days"].fillna(0).astype(float)
    # Log-scale income (right-skewed)
    acc_features["log_income"] = np.log1p(acc_features["monthly_income_zar"])
    acc_features["log_age"] = np.log1p(acc_features["account_age_days"])
    acc_x = torch.tensor(acc_features[["log_income", "log_age"]].values, dtype=torch.float)

    # Account labels: 1 = member of a confirmed fraud ring
    fraud_ring_accounts = set(
        tx[tx["fraud_type"].str.upper().fillna("") == "FRAUD_RING"]["sender_account_id"].unique()
    )
    acc_y = torch.tensor(
        [1 if aid in fraud_ring_accounts else 0 for aid in account_ids],
        dtype=torch.long,
    )

    # Device + merchant nodes: single constant feature (degree computed later)
    dev_x = torch.ones((n_devices, 1), dtype=torch.float)
    mer_x = torch.ones((n_merchants, 1), dtype=torch.float)

    # ── Edge construction ─────────────────────────────────────────────────────

    def _edge(src_ids, dst_ids, src_map, dst_map):
        """Convert lists of string IDs to a [2, E] edge_index tensor."""
        rows, cols = [], []
        for s, d in zip(src_ids, dst_ids):
            if s in src_map and d in dst_map:
                rows.append(src_map[s])
                cols.append(dst_map[d])
        return torch.tensor([rows, cols], dtype=torch.long)

    logger.info("Building edges...")

    # (account, transacts_with, merchant)
    edge_acc_mer = _edge(tx["sender_account_id"], tx["merchant_id"], account_idx, merchant_idx)

    # (account, uses_device, device)
    edge_acc_dev = _edge(tx["sender_account_id"], tx["sender_device_id"], account_idx, device_idx)

    # (device, shared_by, account) — reverse
    edge_dev_acc = edge_acc_dev.flip(0)

    # PayShap peer transfers: (account, payshap_transfer, account)
    payshap_tx = tx[(tx["payment_rail"] == "PAYSHAP") & tx["receiver_account_id"].notna()]
    edge_payshap = _edge(
        payshap_tx["sender_account_id"],
        payshap_tx["receiver_account_id"],
        account_idx,
        account_idx,
    )

    # EFT transfers: (account, eft_transfer, account)
    eft_tx = tx[(tx["payment_rail"] == "EFT") & tx["receiver_account_id"].notna()]
    edge_eft = _edge(
        eft_tx["sender_account_id"],
        eft_tx["receiver_account_id"],
        account_idx,
        account_idx,
    )

    logger.info(
        f"Edges — acc→mer: {edge_acc_mer.shape[1]:,}  "
        f"acc→dev: {edge_acc_dev.shape[1]:,}  "
        f"payshap: {edge_payshap.shape[1]:,}  "
        f"eft: {edge_eft.shape[1]:,}"
    )

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    data["account"].x = acc_x
    data["account"].y = acc_y
    data["account"].node_id = torch.arange(n_accounts)

    data["device"].x = dev_x
    data["merchant"].x = mer_x

    data["account", "transacts_with", "merchant"].edge_index = edge_acc_mer
    data["account", "uses_device", "device"].edge_index = edge_acc_dev
    data["device", "shared_by", "account"].edge_index = edge_dev_acc
    data["account", "payshap_transfer", "account"].edge_index = edge_payshap
    data["account", "eft_transfer", "account"].edge_index = edge_eft

    # ── Sanity check ──────────────────────────────────────────────────────────
    fraud_count = int(acc_y.sum())
    logger.info(
        f"Graph built — {n_accounts:,} account nodes | "
        f"{fraud_count:,} fraud ring members ({100 * fraud_count / n_accounts:.2f}%)"
    )
    logger.info(data)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    logger.info(f"Saved to {output_path}")

    return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions-path", default="data/raw/transactions.parquet")
    parser.add_argument("--accounts-path", default="data/raw/accounts.parquet")
    parser.add_argument("--output-path", default="data/graphs/hetero_graph.pt")
    args = parser.parse_args()

    build_graph(
        transactions_path=args.transactions_path,
        accounts_path=args.accounts_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
