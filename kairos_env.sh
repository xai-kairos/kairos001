conda create -n kairos001 python=3.10 -y
conda activate kairos001

# Base conda packages
conda install -c conda-forge psycopg2 tqdm -y
conda install -y "numpy<2"
pip install scikit-learn networkx xxhash graphviz

# Step 1: Install PyTorch first (M1/ARM macOS version from official source)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Then install PyG + extensions (compatible with torch 2.2.0)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

pip install pytz

 python 4_embedding.py --start "2018-04-06 11:33:00" --end "2018-04-06 11:34:00" --label 4_6_test --skip-daily

python explainer_gnn.py \
  --graph_path ./artifact/graph_embeddings/graph_4_6_test.TemporalData.simple \
  --models_path ./artifact/models/models.pt \
  --target_index 0 \
  --epochs 200 \
  --topk 25

