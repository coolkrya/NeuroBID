# src/inference.py
"""
MVP Inference: Compare two face images using pre-trained MobileFaceNet.
No training required. Runs on MPS/CPU.
"""

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from loguru import logger

from config import load_config
from model import MobileFaceNet, load_pretrained_backbone

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)


def preprocess(image_path: str, device: torch.device) -> torch.Tensor:
    """Resize, normalize and tensorize image for 112x112 face model."""
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def main():
    cfg = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🍎 Device: {device}")

    # 1. Load pre-trained model
    model = MobileFaceNet(embedding_dim=128).to(device)
    weights_path = cfg["paths"].get("pretrained_weights", "models/mobilefacenet.pth")

    if not Path(weights_path).exists():
        logger.error(
            f"❌ Weights not found at `{weights_path}`. Run download command first."
        )
        return

    load_pretrained_backbone(model, weights_path, device)
    model.eval()
    logger.info("✅ Pre-trained MobileFaceNet loaded")

    # 2. Compare two images
    img1 = input("📷 Path to first image: ").strip()
    img2 = input("📷 Path to second image: ").strip()

    if not (Path(img1).exists() and Path(img2).exists()):
        logger.error("❌ One or both images not found.")
        return

    print(preprocess(img1, device).shape)

    with torch.no_grad():
        emb1 = model(preprocess(img1, device))  # [1, 128], L2-normed
        emb2 = model(preprocess(img2, device))  # [1, 128], L2-normed

    # Cosine similarity = dot product for L2-normed vectors
    print(emb1, emb2, sep="\n")
    similarity = torch.mm(emb1, emb2.T).item()
    threshold = 0.45  # Standard for ArcFace/MobileFaceNet on aligned faces

    match = similarity > threshold
    logger.info(f"🔍 Cosine Similarity: {similarity:.4f}")
    logger.info(
        f"👥 Same person: {'✅ YES' if match else '❌ NO'} (threshold: {threshold})"
    )


if __name__ == "__main__":
    main()
