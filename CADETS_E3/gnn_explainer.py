"""
GNNExplainer integration for the CADETS_E3 TGNN model.

This script loads the trained TGN components (memory, gnn, link_pred, neighbor_loader)
and uses PyG's high-level Explainer wrapper with the GNNExplainer algorithm to
explain a *single edge prediction* within a chosen TemporalData graph snapshot.

Notes / assumptions:
- We explain an edge-level prediction (edge type classification) produced by LinkPredictor.
- For stability across the explainer's optimization steps, the model forward here does
  NOT mutate memory or neighbor_loader state. It resets them and performs a single
  feed-forward pass given (edge_index, t, msg) tensors supplied by the caller.
- We only learn an **edge mask** (no node feature masks), because node inputs to the
  GNN are the evolving TGN memory states rather than raw features. If you want to
  attribute node importance, you can switch `node_mask_type` to "object" as well,
  but start with edges for clarity.

Usage (example):
    python gnn_explainer.py \
      --graph_path /Volumes/Projects/Research/KAIROS/kairos001/graphs/graph_4_6.TemporalData.simple \
      --models_path /Volumes/Projects/Research/KAIROS/kairos001/models/models.pt \
      --target_index 0 \
      --epochs 200 

This will print the top-k important edges for the selected target edge and save a
masked subgraph visualization (optional) under ARTIFACT_DIR/explanations/.
"""

import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig, ThresholdConfig

# Project imports
from model import (
    device, TGNMemory, GraphAttentionEmbedding, LinkPredictor, assoc
)
from config import ARTIFACT_DIR, GRAPHS_DIR, node_embedding_dim, include_edge_type
from kairos_utils import ns_time_to_datetime_US


class TGNLinkModel(nn.Module):
  """Wrapper around TGNN components for edge-level classification.
  
  Forward signature is designed to be compatible with PyG Explainer. The explainer
  will inject an edge mask into the internal message passing (TransformerConv), so
  we need to execute the GNN with the provided `edge_index` (and edge attributes).
  """
  def __init__(self, memory: nn.Module, gnn: nn.Module, link_pred: nn.Module):
      super().__init__()
      self.memory = memory
      self.gnn = gnn
      self.link_pred = link_pred

  def forward(
      self,
      x: torch.Tensor,                # unused (we build from memory), kept for API compatibility
      edge_index: torch.Tensor,
      *,
      t: torch.Tensor,                # [num_edges] temporal stamps aligned to edge_index columns
      msg: torch.Tensor,              # [num_edges, msg_dim] edge message attributes
      src: torch.Tensor,              # [num_events] source node indices for edges to classify
      dst: torch.Tensor               # [num_events] destination node indices
  ) -> torch.Tensor:
      # Reset to a deterministic snapshot for every forward call (important for explainer
      # which will call forward many times while optimizing the mask):
      self.memory.reset_state()

      # Build node set from **all** endpoints present in edge_index to avoid OOB indexing
      # in modules that expect edge_index to be **local** indices.
      n_id = edge_index.view(-1).unique()

      # Initialize memory states for involved nodes
      z, last_update = self.memory(n_id)

      # Detach states and attributes so each explainer step builds a fresh graph
      # and we do not try to backprop through a previous step's graph.
      z = z.detach()
      last_update = last_update.detach()
      t = t.detach()
      msg = msg.detach()

      # Map global node indices to local positions in z
      assoc[n_id] = torch.arange(n_id.size(0), device=z.device)

      # Remap global edge_index -> local indices expected by the GNN
      edge_index_local = assoc[edge_index]

      # Run the (masked) GNN over the provided static edge_index with temporal attrs
      # The explainer will hook into MessagePassing to apply an edge_mask internally.
      z = self.gnn(z, last_update, edge_index_local, t, msg)

      # Classify provided (src, dst) pairs
      out = self.link_pred(z[assoc[src]], z[assoc[dst]])
      return out  # [num_events, num_edge_types]


def load_trained(models_path: str) -> Tuple[nn.Module, nn.Module, nn.Module, object]:
    memory, gnn, link_pred, neighbor_loader = torch.load(models_path, map_location='cpu')
    # Ensure proper device move (keep using global `device` from project)
    memory = memory.to(device)
    gnn = gnn.to(device)
    link_pred = link_pred.to(device)
    return memory, gnn, link_pred, neighbor_loader


def run_explainer(graph_path: str, models_path: str, target_index: int, epochs: int, topk: int):
    # Load data and models
    data = torch.load(graph_path, map_location=device)
    memory, gnn, link_pred, _ = load_trained(models_path)

    # Form the wrapper model
    model = TGNLinkModel(memory=memory, gnn=gnn, link_pred=link_pred).to(device)
    model.eval()

    # Build explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type="model",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="edge",
            return_type="raw"  # raw logits
        ),
        node_mask_type=None,               # we don't mask node features here
        edge_mask_type="object",          # learn an edge mask over provided edge_index
        threshold_config=ThresholdConfig(threshold_type="topk", value=topk),
    )

    # Choose a target edge by index in this TemporalData object.
    # We treat each (src[i], dst[i]) as an event to classify.
    num_events = data.src.size(0)
    if target_index < 0 or target_index >= num_events:
        raise IndexError(f"target_index {target_index} is out of range [0, {num_events-1}]")

    # Slice tensors for a minimal-but-sane batch around the target edge.
    # For simplicity, we explain *only the single edge* at target_index. You can
    # extend to a window of edges if desired.
    src = data.src[target_index:target_index+1].to(device)
    dst = data.dst[target_index:target_index+1].to(device)

    # For the GNN forward, we still need an edge set to propagate over. Here we
    # use all edges present in `data` (static snapshot explanation). If you prefer
    # a local neighborhood, filter `edge_index`, `t`, and `msg` first.
    # Convert TemporalData edges into COO `edge_index` along with aligned attrs.
    # In this codebase, `data` is already preprocessed similar to training usage.
    edge_index = torch.stack([data.src, data.dst], dim=0).to(device)
    t = data.t.to(device)
    msg = data.msg.to(device)

    # Dummy x placeholder to satisfy the Explainer API; not used by TGNLinkModel.
    # Shape must match the number of nodes participating, but it's ignored.
    # We pass a small tensor and ignore its content in forward.
    x = torch.zeros(1, 1, device=device)

    # Run explanation. `index=0` selects our single edge in the model output.
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        index=0,
        t=t,
        msg=msg,
        src=src,
        dst=dst,
    )

    # Report
    print("Target edge (global ids):", int(src[0].item()), "->", int(dst[0].item()))
    print("Predicted logits:", explanation.prediction)

    # Extract and display top edges by importance
    edge_mask = explanation.edge_mask
    if edge_mask is not None:
        topk_vals, topk_idx = torch.topk(edge_mask, k=min(topk, edge_mask.numel()))
        print("Top-{} important edges (index: importance):".format(topk))
        for rank in range(topk_vals.numel()):
            eidx = int(topk_idx[rank])
            importance = float(topk_vals[rank])
            s = int(edge_index[0, eidx])
            d = int(edge_index[1, eidx])
            ts = ns_time_to_datetime_US(int(t[eidx])) if t is not None else "-"
            print(f"  #{rank+1}: ({s} -> {d}) @ {ts}  score={importance:.4f}")

    # Optional: save masks/subgraph artifacts
    out_dir = os.path.join(ARTIFACT_DIR, "explanations")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(explanation, os.path.join(out_dir, f"gnnexplainer_edge_{int(src[0])}_{int(dst[0])}.pt"))
    print(f"Saved explanation object to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_path", type=str, default=os.path.join(GRAPHS_DIR, "graph_4_6.TemporalData.simple"))
    p.add_argument("--models_path", type=str, default=os.path.join(os.path.dirname(GRAPHS_DIR), "models", "models.pt"))
    p.add_argument("--target_index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--topk", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_explainer(
        graph_path=args.graph_path,
        models_path=args.models_path,
        target_index=args.target_index,
        epochs=args.epochs,
        topk=args.topk,
    )
