# debug_similarity.py
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

from src.config import load_config
from src.model import MobileFaceNet, load_pretrained_backbone
from torchvision import transforms
from PIL import Image
import torch

cfg = load_config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load model
model = MobileFaceNet(embedding_dim=128).to(device)
load_pretrained_backbone(model, cfg["paths"]["pretrained_weights"], device)
model.eval()

# 2. Exact same pipeline as extract/search
transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# 3. Query image (change if needed)
query_path = (
    sys.argv[1] if len(sys.argv) > 1 else "data/processed/1000_persons/val/13/0000.jpg"
)
img = Image.open(query_path).convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    query_emb = model(tensor).cpu().numpy().flatten()

# 4. Direct check against index
emb_path = Path(cfg["paths"]["exp_dir"]) / "embeddings.npy"
if not emb_path.exists():
    print("❌ embeddings.npy not found. Run `python rebuild_index.py` first.")
    sys.exit(1)

db_embs = np.load(emb_path).astype(np.float32)
# Normalize just in case
query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
db_norms = np.linalg.norm(db_embs, axis=1, keepdims=True)
db_embs_normed = db_embs / (db_norms + 1e-9)

sims = db_embs_normed @ query_emb
max_sim = sims.max()
best_idx = sims.argmax()

print(f"🔍 Query: {query_path}")
print(f"📊 Max similarity (numpy direct): {max_sim:.6f}")
print(f"📍 Best index in DB: {best_idx}")

# Check metadata
import csv

meta_path = Path(cfg["paths"]["exp_dir"]) / "metadata.csv"
if meta_path.exists():
    with open(meta_path, "r") as f:
        reader = list(csv.DictReader(f))
        if best_idx < len(reader):
            print(
                f"📄 Matched file: {reader[best_idx].get('relative_path', 'unknown')}"
            )
        else:
            print("⚠️ Index out of bounds! metadata.csv vs embeddings.npy mismatch.")

if max_sim < 0.90:
    print("\n⚠️ CRITICAL: Model/Index mismatch. Rebuild index or check transforms.")
else:
    print(
        "\n✅ Model & Index are healthy. Issue is in search.py threshold or FAISS call."
    )
