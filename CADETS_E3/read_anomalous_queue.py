#!/usr/bin/env python3
import os
import torch
import pandas as pd
from config import ARTIFACT_DIR

# 1) Point this at your history_list files directory
HISTORY_DIR = ARTIFACT_DIR

# 2) Gather all files named like "*_history_list"
files = sorted(f for f in os.listdir(HISTORY_DIR)
               if f.endswith("_history_list"))

all_rows = []

for fname in files:
    path = os.path.join(HISTORY_DIR, fname)
    # 3) Load the saved history_list (list of queues)
    history_list = torch.load(path)
    
    # 4) Flatten into a row per time-window per queue
    for queue_id, queue in enumerate(history_list):
        for tw in queue:
            all_rows.append({
                "file": fname,
                "queue": queue_id,
                "index": tw.get("index"),
                "name": tw.get("name"),
                "loss": tw.get("loss"),
                # join nodeset into a semicolon-separated string
                "nodeset": ";".join(map(str, sorted(tw.get("nodeset", []))))
            })

# 5) Build a DataFrame
df = pd.DataFrame(all_rows)

# 6) Export to CSV
csv_path = os.path.join(HISTORY_DIR, "all_history_list.csv")
df.to_csv(csv_path, index=False)
print(f"Wrote combined CSV to {csv_path}")

# 7) Export to TSV (easier to view in terminal)
tsv_path = os.path.join(HISTORY_DIR, "all_history_list.tsv")
df.to_csv(tsv_path, sep="\t", index=False)
print(f"Wrote combined TSV to {tsv_path}")

# 8) Print a preview in the terminal
pd.set_option("display.max_rows", None, "display.max_columns", None)
print("\n=== Preview of loaded history lists ===\n")
print(df.head(20).to_string(index=False))