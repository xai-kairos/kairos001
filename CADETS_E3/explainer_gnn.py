"""Lightweight edge explainer for the CADETS_E3 temporal GNN."""

import argparse
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import TemporalData
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig, ThresholdConfig

from model import device, GraphAttentionEmbedding, LinkPredictor, TGNMemory, assoc
from config import ARTIFACT_DIR, GRAPHS_DIR, include_edge_type, node_embedding_dim
from kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US, tensor_find


class TGNLinkModel(nn.Module):
    """Wrapper that runs the trained TGNN components on a static snapshot."""

    def __init__(self, memory: nn.Module, gnn: nn.Module, link_pred: nn.Module):
        super().__init__()
        self.memory = memory
        self.gnn = gnn
        self.link_pred = link_pred

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        t: torch.Tensor,
        msg: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        # Always start from a clean snapshot for the explainer optimisation.
        self.memory.reset_state()

        # Prepare node features for all unique nodes that appear in this snapshot.
        n_id = edge_index.view(-1).unique()
        z, last_update = self.memory(n_id)

        # Keep all tensors detached so each explainer step is independent.
        z = z.detach()
        last_update = last_update.detach()
        t = t.detach()
        msg = msg.detach()

        assoc[n_id] = torch.arange(n_id.size(0), device=z.device)
        edge_index_local = assoc[edge_index]
        z = self.gnn(z, last_update, edge_index_local, t, msg)
        out = self.link_pred(z[assoc[src]], z[assoc[dst]])
        return out


def load_trained(models_path: str):
    memory, gnn, link_pred, neighbor_loader = torch.load(models_path, map_location="cpu")
    memory = memory.to(device)
    gnn = gnn.to(device)
    link_pred = link_pred.to(device)
    return memory, gnn, link_pred, neighbor_loader


def _parse_time(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if '.' in value:
        # Remove fractional seconds produced by ns_time_to_datetime_US
        value = value.split('.', 1)[0]
    if value.isdigit():
        val = int(value)
        if len(value) <= 10:  # seconds -> nanoseconds
            return val * 1_000_000_000
        return val
    digits = ''.join(ch for ch in value if ch.isdigit())
    if len(digits) == 12:  # YYYYMMDDHHMM
        date = digits[:8]
        time = digits[8:]
        return datetime_to_ns_time_US(f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:]}:00")
    if len(digits) == 14:  # YYYYMMDDHHMMSS
        date = digits[:8]
        time = digits[8:]
        return datetime_to_ns_time_US(f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}:{time[4:]}")
    return datetime_to_ns_time_US(value)


def _find_target_index(data: TemporalData,
                       src: Optional[int],
                       dst: Optional[int],
                       timestamp_ns: Optional[int],
                       fallback: int) -> int:
    if src is None or dst is None:
        return fallback
    mask = (data.src == src) & (data.dst == dst)
    if timestamp_ns is not None:
        mask &= data.t == timestamp_ns
    indices = torch.where(mask)[0]
    if indices.numel() == 0:
        raise ValueError("Requested target edge not present in the graph snapshot.")
    return int(indices[0].item())


def _summarise_edges(edge_mask: torch.Tensor,
                     edge_index: torch.Tensor,
                     timestamps: torch.Tensor,
                     topk: int) -> List[Dict[str, object]]:
    if edge_mask.numel() == 0:
        return []
    k = min(topk, edge_mask.numel())
    vals, idx = torch.topk(edge_mask, k=k)
    out: List[Dict[str, object]] = []
    total = edge_mask.sum().item() or 1.0
    for rank in range(k):
        eidx = int(idx[rank])
        importance = float(vals[rank])
        share = importance / total * 100.0
        src = int(edge_index[0, eidx])
        dst = int(edge_index[1, eidx])
        ts_ns = int(timestamps[eidx])
        out.append(
            {
                "rank": rank + 1,
                "src": src,
                "dst": dst,
                "timestamp_ns": ts_ns,
                "timestamp": ns_time_to_datetime_US(ts_ns),
                "score": importance,
                "share": share,
            }
        )
    return out


def _true_label(msg_row: torch.Tensor) -> Optional[int]:
    if msg_row is None:
        return None
    slice_ = msg_row[node_embedding_dim:-node_embedding_dim]
    try:
        idx = tensor_find(slice_.cpu(), 1) - 1
    except Exception:
        idx = int(torch.argmax(slice_).item())
    return idx if idx >= 0 else None


def run_explainer(
    graph_path: str,
    models_path: str,
    target_index: int,
    epochs: int,
    topk: int,
    *,
    target_src: Optional[int] = None,
    target_dst: Optional[int] = None,
    target_time: Optional[str] = None,
) -> Dict[str, object]:
    data: TemporalData = torch.load(graph_path, map_location="cpu")

    timestamp_ns = _parse_time(target_time)
    resolved_index = _find_target_index(data, target_src, target_dst, timestamp_ns, target_index)
    src_val = int(data.src[resolved_index])
    dst_val = int(data.dst[resolved_index])
    time_val = int(data.t[resolved_index])

    memory, gnn, link_pred, _ = load_trained(models_path)
    model = TGNLinkModel(memory=memory, gnn=gnn, link_pred=link_pred).to(device)
    model.eval()

    edge_index = torch.stack([data.src, data.dst], dim=0).to(device)
    t = data.t.to(device)
    msg = data.msg.to(device)
    src_tensor = data.src[resolved_index:resolved_index + 1].to(device)
    dst_tensor = data.dst[resolved_index:resolved_index + 1].to(device)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type="model",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="edge",
            return_type="raw",
        ),
        node_mask_type=None,
        edge_mask_type="object",
        threshold_config=ThresholdConfig(threshold_type="topk", value=topk),
    )

    logits = model(torch.zeros(1, 1, device=device), edge_index, t=t, msg=msg, src=src_tensor, dst=dst_tensor)
    probs = torch.softmax(logits[0], dim=-1)
    pred_idx = int(torch.argmax(probs).item())
    pred_label = include_edge_type[pred_idx] if pred_idx < len(include_edge_type) else f"idx_{pred_idx}"
    true_idx = _true_label(data.msg[resolved_index])
    true_label = include_edge_type[true_idx] if (true_idx is not None and true_idx < len(include_edge_type)) else "unknown"

    explanation = explainer(
        x=torch.zeros(1, 1, device=device),
        edge_index=edge_index,
        index=0,
        t=t,
        msg=msg,
        src=src_tensor,
        dst=dst_tensor,
    )

    edge_mask = explanation.edge_mask.detach().cpu()
    timestamps_cpu = data.t.cpu()
    edge_index_cpu = edge_index.cpu()
    top_edges = _summarise_edges(edge_mask, edge_index_cpu, timestamps_cpu, topk)

    print(f"Target edge: {src_val} -> {dst_val} @ {ns_time_to_datetime_US(time_val)}")
    print(f"  True={true_label}  Pred={pred_label}  Prob={float(probs[pred_idx]):.4f}")
    for item in top_edges:
        print(f"  #{item['rank']}: ({item['src']} -> {item['dst']}) score={item['score']:.4f}")

    out_dir = os.path.join(ARTIFACT_DIR, "explanations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gnnexplainer_edge_{src_val}_{dst_val}_{time_val}.pt")
    torch.save(explanation, out_path)

    return {
        "target": {
            "src": src_val,
            "dst": dst_val,
            "timestamp_ns": time_val,
            "timestamp": ns_time_to_datetime_US(time_val),
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_prob": float(probs[pred_idx]),
        },
        "top_edges": top_edges,
        "explanation_path": out_path,
        "edge_mask": edge_mask,
        "edge_index": edge_index_cpu,
        "timestamps": timestamps_cpu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a single TGNN edge prediction.")
    parser.add_argument("--graph_path", type=str, default=os.path.join(GRAPHS_DIR, "graph_4_6.TemporalData.simple"))
    parser.add_argument("--models_path", type=str, default=os.path.join(os.path.dirname(GRAPHS_DIR), "models", "models.pt"))
    parser.add_argument("--target_index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--target_src", type=int)
    parser.add_argument("--target_dst", type=int)
    parser.add_argument("--target_time", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_explainer(
        graph_path=args.graph_path,
        models_path=args.models_path,
        target_index=args.target_index,
        epochs=args.epochs,
        topk=args.topk,
        target_src=args.target_src,
        target_dst=args.target_dst,
        target_time=args.target_time,
    )
