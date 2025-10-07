import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from config import ARTIFACT_DIR, include_edge_type
from kairos_utils import ns_time_to_datetime_US

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency guard
    plt = None
    nx = None

DEFAULT_PATTERN = "gnnexplainer_edge_*.pt"


def _fetch_attr(obj, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    try:
        return obj.get(name, default)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        return obj[name]  # type: ignore[index]
    except Exception:
        return default


def _resolve_paths(paths: Sequence[str], pattern: str) -> List[Path]:
    candidates: List[Path] = []
    if not paths:
        candidates = sorted((Path(ARTIFACT_DIR) / "explanations").glob(pattern))
    else:
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                candidates.extend(sorted(p.glob(pattern)))
            else:
                candidates.append(p)
    seen = set()
    ordered: List[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _edge_summaries(edge_index: torch.Tensor, edge_mask: torch.Tensor, topk: int,
                    timestamps: Optional[torch.Tensor]) -> List[Tuple[int, int, float, Optional[str]]]:
    if edge_mask.numel() == 0:
        return []
    k = edge_mask.numel() if topk <= 0 else min(topk, edge_mask.numel())
    vals, idx = torch.topk(edge_mask, k=k)
    summaries: List[Tuple[int, int, float, Optional[str]]] = []
    for rank in range(k):
        eidx = int(idx[rank])
        src = int(edge_index[0, eidx])
        dst = int(edge_index[1, eidx])
        score = float(vals[rank])
        ts_str: Optional[str] = None
        if timestamps is not None:
            ts_str = ns_time_to_datetime_US(int(timestamps[eidx]))
        summaries.append((src, dst, score, ts_str))
    return summaries


def _prediction_summary(pred: torch.Tensor) -> Tuple[List[float], int, Optional[str]]:
    pred = pred.detach().cpu().squeeze()
    probs = torch.softmax(pred, dim=-1)
    pred_idx = int(torch.argmax(pred).item())
    label = include_edge_type[pred_idx] if pred_idx < len(include_edge_type) else None
    return probs.tolist(), pred_idx, label


def _draw_graph(path: Path, edge_index: torch.Tensor, edge_mask: torch.Tensor,
                save_dir: Optional[Path], show_plot: bool) -> None:
    if nx is None or plt is None:
        print("[warn] networkx/matplotlib not available; skipping plot.")
        return
    g = nx.DiGraph()
    edge_mask = edge_mask.detach().cpu()
    edge_index = edge_index.detach().cpu()
    edges = list(edge_index.t().tolist())
    for src, dst in edges:
        g.add_edge(src, dst)
    if g.number_of_edges() == 0:
        print("[warn] no edges to draw for", path)
        return
    pos = nx.spring_layout(g, seed=0)
    weights = edge_mask.tolist()
    if weights:
        min_w = min(weights)
        max_w = max(weights)
        span = max(max_w - min_w, 1e-6)
        widths = [1.0 + 4.0 * ((w - min_w) / span) for w in weights]
    else:
        widths = [1.0] * g.number_of_edges()
    cmap = plt.cm.viridis
    normed = []
    if weights:
        min_w = min(weights)
        max_w = max(weights)
        span = max(max_w - min_w, 1e-6)
        normed = [(w - min_w) / span for w in weights]
    colors = [cmap(v if normed else 0.5) for v in (normed if normed else [0.5] * len(weights))]
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(g, pos, node_size=600, node_color="#1f77b4")
    nx.draw_networkx_labels(g, pos, font_size=8, font_color="white")
    nx.draw_networkx_edges(g, pos, edgelist=edges, arrows=True, edge_color=colors, width=widths)
    plt.title(path.name)
    plt.axis("off")
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{path.stem}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {out_path}")
    if show_plot:
        plt.show()
    plt.close()


def process_file(path: Path, topk: int, save_dir: Optional[Path], show_plot: bool) -> None:
    explanation = torch.load(path, map_location="cpu")
    print(f"=== {path} ===")
    prediction = _fetch_attr(explanation, "prediction")
    if prediction is None:
        print("No prediction found in explanation.")
    else:
        probs, pred_idx, label = _prediction_summary(prediction)
        label_str = label if label is not None else f"index {pred_idx}"
        print(f"Predicted class: {label_str} (idx={pred_idx})")
        print(f"Probabilities: {[round(p, 4) for p in probs]}")
    edge_index = _fetch_attr(explanation, "edge_index")
    if edge_index is None:
        edge_index = _fetch_attr(explanation, "edge_label_index")
    edge_mask = _fetch_attr(explanation, "edge_mask")
    if edge_index is None or edge_mask is None:
        print("No edge_index/edge_mask stored; cannot unpack edges.")
        return
    timestamps = _fetch_attr(explanation, "t")
    if timestamps is None:
        timestamps = _fetch_attr(explanation, "timestamps")
    if timestamps is None:
        timestamps = _fetch_attr(explanation, "edge_t")
    summaries = _edge_summaries(edge_index, edge_mask, topk, timestamps)
    if not summaries:
        print("Edge mask is empty.")
    else:
        print(f"Top {len(summaries)} edges by importance:")
        for rank, (src, dst, score, ts) in enumerate(summaries, start=1):
            ts_part = f" @ {ts}" if ts else ""
            print(f"  {rank:>2}. {src} -> {dst}{ts_part}  score={score:.4f}")
    _draw_graph(path, edge_index, edge_mask, save_dir, show_plot)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved GNNExplainer artifacts.")
    parser.add_argument("paths", nargs="*", help="Paths or directories of explanation .pt files.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob to match when a directory is given.")
    parser.add_argument("--topk", type=int, default=10, help="How many top edges to report.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Optional directory to store rendered plots.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    save_dir = Path(args.save_dir) if args.save_dir else None
    paths = _resolve_paths(args.paths, args.pattern)
    if not paths:
        print("No explanation files found.")
        return
    for path in paths:
        process_file(path, topk=args.topk, save_dir=save_dir, show_plot=args.show)


if __name__ == "__main__":
    main()
