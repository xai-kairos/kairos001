import os
import torch
from torch_geometric.data.temporal import TemporalData
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([TemporalData, GlobalStorage])
import pandas as pd
# from sqlalchemy import create_engine

EMBED_DIR = "artifact/graph_embeddings/"
files = sorted(f for f in os.listdir(EMBED_DIR)
               if f.endswith(".TemporalData.simple"))

for fname in files:
    path = os.path.join(EMBED_DIR, fname)
    data = torch.load(path, weights_only=True)  # safer going forward
    # get your core tensors
    src, dst, t, msg = data.src, data.dst, data.t, data.msg
    print(f"\n=== {fname} ===")
    print(f"Edges: {src.size(0)}  |  Features: {msg.size(1)}-dim")
    print(" src:", src[:5].tolist())
    print(" dst:", dst[:5].tolist())
    print("  t :", t[:5].tolist())
    print("msg shape:", msg[:5].shape)
    # Compute and report unique nodes across src and dst
    unique_nodes = set(src.tolist()) | set(dst.tolist())
    print(f"Unique nodes (src + dst): {len(unique_nodes)}")

    # Optional: turn into pandas for deeper exploration
    import pandas as pd
    df = pd.DataFrame({
        "src": src.tolist(),
        "dst": dst.tolist(),
        "t": t.tolist(),
        **{f"feat_{i}": msg[:, i].tolist() for i in range(msg.size(1))}
    })
    print(df.head())
    
    # Build DataFrame of edges and features
    df = pd.DataFrame({
        "src": src.tolist(),
        "dst": dst.tolist(),
        "t": t.tolist(),
        **{f"feat_{i}": msg[:, i].tolist() for i in range(msg.size(1))}
    })
    
    # Export to CSV
    csv_path = os.path.join(EMBED_DIR, fname.replace(".TemporalData.simple", ".csv"))
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV to {csv_path}")
    
    # # Optional: Export to PostgreSQL (requires SQLAlchemy and a running DB)
    # engine = create_engine("postgresql://username:password@localhost:5432/your_db")
    # table_name = fname.replace(".TemporalData.simple", "").replace(".", "_")
    # df.to_sql(table_name, engine, if_exists="replace", index=False)
    # print(f"Wrote table '{table_name}' to Postgres")