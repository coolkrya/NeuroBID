# test_model_consistency.py
import torch, sys
from pathlib import Path

sys.path.insert(0, ".")

from src.config import load_config
from src.model import MobileFaceNet, load_pretrained_backbone
from torchvision import transforms
from PIL import Image
from loguru import logger


logger.remove()

]
logger.add(sys.stderr, format="{level} | {name}:{line} - {message}")

cfg = load_config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = MobileFaceNet(embedding_dim=128).to(device).eval()
load_pretrained_backbone(model, cfg["paths"]["pretrained_weights"], device)


transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


img = Image.open("data/processed/1000_persons/train/13/0000.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    emb1 = model(tensor)
    emb2 = model(tensor)  
    sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    norm = emb1.norm().item()

print(f"🔍 Self-similarity (should be 1.0): {sim:.6f}")
print(f"📏 L2 norm (should be 1.0): {norm:.6f}")
print(
    f"📊 Embedding stats: min={emb1.min():.4f}, max={emb1.max():.4f}, mean={emb1.mean():.4f}"
)

if sim < 0.99:
    print("\n❌ CRITICAL: Model is unstable or weights not loaded properly.")
elif norm < 0.99:
    print("\n❌ CRITICAL: L2 normalization in forward() is not working.")
else:
    print("\n✅ Model is healthy. ")
