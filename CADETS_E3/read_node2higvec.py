#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
import sys
sys.modules['numpy._core'] = np.core



# --- CONFIG ---
GRAPHS_DIR       = "artifact/graphs"
NODE_VEC_PATH    = "artifact/node2higvec.npy"    # adjust if you used a .pt/.pth suffix
DEVICE           = "cpu"                     # or "cuda" if appropriate
RTOL, ATOL       = 1e-6, 1e-6                # tolerance for allclose checks

def load_node_embeddings(path):
    """Load the node2higvec array and return a numpy array."""
    if path.endswith(".npy"):
        return np.load(path)
    # otherwise fallback to torch.load
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    raise ValueError(f"Unexpected format in {path}")

def verify_graph(file_path, node_higs):
    data = torch.load(file_path, map_location=DEVICE)
    src = data.src.cpu().numpy()
    dst = data.dst.cpu().numpy()
    msg = data.msg.cpu().numpy()
    
    D = node_higs.shape[1]
    if msg.shape[1] < 2*D:
        raise ValueError(f"{file_path}: msg dim {msg.shape[1]} too small for D={D}")
    
    # split out the src- and dst- embeddings
    src_embs = msg[:, :D]
    dst_embs = msg[:, -D:]
    
    # look for mismatches
    bad = []
    for i in range(msg.shape[0]):
        sid, did = src[i], dst[i]
        if not np.allclose(src_embs[i], node_higs[sid], rtol=RTOL, atol=ATOL):
            bad.append((i, "src", sid))
        if not np.allclose(dst_embs[i], node_higs[did], rtol=RTOL, atol=ATOL):
            bad.append((i, "dst", did))
    return bad

def main():
    node_higs = load_node_embeddings(NODE_VEC_PATH)
    pattern  = os.path.join(GRAPHS_DIR, "*.TemporalData.simple")
    files    = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        return

    total_bad = 0
    for f in files:
        print(f"\nVerifying {os.path.basename(f)} …")
        bad = verify_graph(f, node_higs)
        if not bad:
            print("  ✓ all embeddings match")
        else:
            print(f"  ✗ {len(bad)} mismatches:")
            for idx, role, nid in bad[:10]:
                print(f"    edge #{idx}: {role} node {nid}")
            if len(bad) > 10:
                print(f"    … and {len(bad)-10} more")
            total_bad += len(bad)

    if total_bad == 0:
        print("\nAll graphs verified successfully.")
    else:
        print(f"\nTotal mismatches across all files: {total_bad}")

if __name__ == "__main__":
    main()