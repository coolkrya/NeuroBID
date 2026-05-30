# test_direct.py
import numpy as np, torch, sys
from pathlib import Path

sys.path.insert(0, ".")
from src.config import load_config
from src.model import MobileFaceNet, load_pretrained_backbone
from torchvision import transforms
from PIL import Image

from loguru import logger


logger.remove()


logger.add(sys.stderr, format="{level} | {name}:{line} - {message}")

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

cfg = load_config()
device = torch.device("mps")
model = MobileFaceNet(embedding_dim=128).to(device)
model.eval()
load_pretrained_backbone(model, cfg["paths"]["pretrained_weights"], device)

transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# Загрузите базу и запрос
query1 = (
    transform(
        Image.open("data/processed/1000_persons/train/13/0000.jpg").convert("RGB")
    )
    .unsqueeze(0)
    .to(device)
)
query2 = (
    transform(Image.open("data/processed/1000_persons/val/13/0009.jpg").convert("RGB"))
    .unsqueeze(0)
    .to(device)
)
with torch.no_grad():
    q_emb1 = model(query1).cpu().numpy().flatten()
    q_emb2 = model(query2).cpu().numpy().flatten()

db = np.load("experiments/exp_001/embeddings.npy").astype(np.float32)
# Нормализуем базу на лету для теста
"""print(db[0])"""
print("Model processed rn - ", q_emb1)

print("Taken from the index - ", q_emb1)


sims = db @ q_emb1
# print(sims)
print(f"🔍 Direct numpy max sim: {sims.max():.4f} at index {sims.argmax()}")

