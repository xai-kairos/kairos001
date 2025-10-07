"""Minimal pipeline: build window graph, score edges, run explainer on top-K losses."""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import importlib
import torch
from torch_geometric.loader import TemporalDataLoader

try:  # when executed from project root
    from config import ARTIFACT_DIR, GRAPHS_DIR, MODELS_DIR, BATCH, node_embedding_dim
    from kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US, init_database_connection, tensor_find
    from model import assoc, device
    import explainer_gnn as explainer_gnn
    import explainer_pg as explainer_pg
    embedding_mod = importlib.import_module("4_embedding")
except ModuleNotFoundError:  # when executed from inside CADETS_E3/
    from CADETS_E3.config import ARTIFACT_DIR, GRAPHS_DIR, MODELS_DIR, BATCH, node_embedding_dim
    from CADETS_E3.kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US, init_database_connection, tensor_find
    from CADETS_E3.model import assoc, device
    import CADETS_E3.explainer_gnn as explainer_gnn
    import CADETS_E3.explainer_pg as explainer_pg
    embedding_mod = importlib.import_module("CADETS_E3.4_embedding")


def _safe_label(raw: str) -> str:
    return ''.join(ch if (ch.isalnum() or ch in {'_', '-', '.'}) else '_' for ch in raw)


def _load_or_create_features(cur):
    node_path = os.path.join(ARTIFACT_DIR, "node2higvec")
    rel_path = os.path.join(ARTIFACT_DIR, "rel2vec")
    if os.path.exists(node_path):
        node2higvec = torch.load(node_path)
    else:
        node2higvec = embedding_mod.gen_feature(cur=cur)
    if os.path.exists(rel_path):
        rel2vec = torch.load(rel_path)
    else:
        rel2vec = embedding_mod.gen_relation_onehot()
    return node2higvec, rel2vec


def ensure_window_graph(start_ns: int, end_ns: int, label: str, force: bool = False) -> str:
    graph_name = _safe_label(label)
    out_path = os.path.join(GRAPHS_DIR, f"graph_{graph_name}.TemporalData.simple")
    if not force and os.path.exists(out_path):
        return out_path

    cur, conn = init_database_connection()
    try:
        node2higvec, rel2vec = _load_or_create_features(cur)
        embedding_mod.gen_vectorized_window(
            cur=cur,
            node2higvec=node2higvec,
            rel2vec=rel2vec,
            logger=embedding_mod.logger,
            start_ts=start_ns,
            end_ts=end_ns,
            label=graph_name,
        )
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    if not os.path.exists(out_path):
        raise RuntimeError("Failed to generate graph window")
    return out_path


@dataclass
class EdgeRecord:
    index: int
    src: int
    dst: int
    timestamp_ns: int
    loss: float
    true_label: int


def _summarise_mask(edge_mask: torch.Tensor,
                    edge_index: torch.Tensor,
                    timestamps: torch.Tensor,
                    topk: int):
    if edge_mask is None or edge_mask.numel() == 0:
        return []
    k = min(topk, edge_mask.numel())
    vals, idx = torch.topk(edge_mask, k=k)
    total = edge_mask.sum().item() or 1.0
    result = []
    for rank in range(k):
        eidx = int(idx[rank])
        importance = float(vals[rank])
        share = importance / total * 100.0
        src = int(edge_index[0, eidx])
        dst = int(edge_index[1, eidx])
        ts_ns = int(timestamps[eidx])
        result.append({
            "rank": rank + 1,
            "src": src,
            "dst": dst,
            "timestamp_ns": ts_ns,
            "timestamp": ns_time_to_datetime_US(ts_ns),
            "score": importance,
            "share": share,
        })
    return result


def score_edges(data, memory, gnn, link_pred, neighbor_loader) -> List[EdgeRecord]:
    memory.eval()
    gnn.eval()
    link_pred.eval()

    memory.reset_state()
    if neighbor_loader is not None:
        neighbor_loader.reset_state()

    loader = TemporalDataLoader(data, batch_size=BATCH, shuffle=False)
    results: List[EdgeRecord] = []
    offset = 0

    for batch in loader:
        src = batch.src.to(device)
        dst = batch.dst.to(device)
        t = batch.t.to(device)
        msg = batch.msg.to(device)

        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        logits = link_pred(z[assoc[src]], z[assoc[dst]])

        true_idx = []
        for row in msg:
            slice_ = row[node_embedding_dim:-node_embedding_dim].detach().cpu()
            try:
                label = tensor_find(slice_, 1) - 1
            except Exception:
                label = int(torch.argmax(slice_).item())
            true_idx.append(label)
        true_tensor = torch.tensor(true_idx, device=device, dtype=torch.long)
        losses = torch.nn.functional.cross_entropy(logits, true_tensor, reduction='none')

        for i in range(src.size(0)):
            record = EdgeRecord(
                index=offset + i,
                src=int(src[i].item()),
                dst=int(dst[i].item()),
                timestamp_ns=int(t[i].item()),
                loss=float(losses[i].item()),
                true_label=int(true_idx[i]),
            )
            results.append(record)

        memory.update_state(src, dst, t, msg)
        if neighbor_loader is not None:
            neighbor_loader.insert(src, dst)
        offset += src.size(0)

    results.sort(key=lambda rec: rec.loss, reverse=True)
    return results


def run_pipeline(timestamp: str,
                 window_minutes: int,
                 topk: int,
                 explain_epochs: int,
                 explain_topk: int,
                 explainer_name: str,
                 force_regen: bool,
                 output_dir: Optional[str]) -> dict:
    ts = timestamp.strip()
    if ts and len(ts) == 16:  # YYYY-MM-DD HH:MM
        ts = ts + ":00"
    start_ns = datetime_to_ns_time_US(ts)
    end_ns = start_ns + window_minutes * 60 * 1_000_000_000

    label = f"{timestamp.replace(' ', '_')}_{window_minutes}m"
    graph_path = ensure_window_graph(start_ns, end_ns, label, force=force_regen)

    data = torch.load(graph_path, map_location='cpu')
    if explainer_name == "pg":
        load_fn = explainer_pg.load_trained
        explainer_module = explainer_pg
    else:
        load_fn = explainer_gnn.load_trained
        explainer_module = explainer_gnn

    memory, gnn, link_pred, neighbor_loader = load_fn(os.path.join(MODELS_DIR, "models.pt"))

    edge_scores = score_edges(data, memory, gnn, link_pred, neighbor_loader)
    top_edges = edge_scores[:topk]

    explanations = []
    aggregate_mask = None
    edge_index_ref = None
    timestamps_ref = None

    for rec in top_edges:
        summary = explainer_module.run_explainer(
            graph_path=graph_path,
            models_path=os.path.join(MODELS_DIR, "models.pt"),
            target_index=rec.index,
            epochs=explain_epochs,
            topk=explain_topk,
            target_src=rec.src,
            target_dst=rec.dst,
            target_time=str(rec.timestamp_ns),
        )
        edge_mask = summary.pop("edge_mask", None)
        edge_index_cpu = summary.pop("edge_index", None)
        timestamps_cpu = summary.pop("timestamps", None)
        if edge_mask is not None:
            if aggregate_mask is None:
                aggregate_mask = edge_mask.clone()
                edge_index_ref = edge_index_cpu
                timestamps_ref = timestamps_cpu
            else:
                aggregate_mask = aggregate_mask + edge_mask
        summary["loss"] = rec.loss
        explanations.append(summary)

    aggregate_summary = None
    if aggregate_mask is not None and edge_index_ref is not None and timestamps_ref is not None:
        aggregate_summary = {
            "total_importance": float(aggregate_mask.sum().item()),
            "top_edges": _summarise_mask(aggregate_mask, edge_index_ref, timestamps_ref, explain_topk),
            "count": len(explanations),
        }

    result = {
        "timestamp": timestamp,
        "window_minutes": window_minutes,
        "graph_path": graph_path,
        "top_edges": [
            {
                "src": rec.src,
                "dst": rec.dst,
                "timestamp_ns": rec.timestamp_ns,
                "timestamp": ns_time_to_datetime_US(rec.timestamp_ns),
                "loss": rec.loss,
                "true_label": rec.true_label,
            }
            for rec in top_edges
        ],
        "explanations": explanations,
        "aggregate_explanation": aggregate_summary,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"xai_summary_{_safe_label(label)}.json")
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        result["summary_path"] = out_file

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run minimal XAI pipeline over attack timestamps.")
    parser.add_argument("timestamp", nargs="+", help="Timestamp(s) in 'YYYY-MM-DD HH:MM[:SS]' format.")
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--explain-epochs", type=int, default=200)
    parser.add_argument("--explain-topk", type=int, default=10)
    parser.add_argument("--explainer", choices=["gnn", "pg"], default="gnn", help="Explainer algorithm to use.")
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--output-dir", type=str, default=os.path.join(ARTIFACT_DIR, "explanations"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for ts in args.timestamp:
        summary = run_pipeline(
            timestamp=ts,
            window_minutes=args.window_minutes,
            topk=args.topk,
            explain_epochs=args.explain_epochs,
            explain_topk=args.explain_topk,
            explainer_name=args.explainer,
            force_regen=args.force_regen,
            output_dir=args.output_dir,
        )
        print("\nCompleted:", ts)
        print("  Graph:", summary["graph_path"])
        if "summary_path" in summary:
            print("  Summary JSON:", summary["summary_path"])
