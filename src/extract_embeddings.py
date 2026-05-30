# src/extract_embeddings.py
"""
Extract face embeddings using the EXACT same pipeline as inference.py.
No batching quirks, no hidden transforms — just pure replication.
"""

import torch
import numpy as np
import csv
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from loguru import logger
from tqdm import tqdm

from config import load_config
from model import MobileFaceNet, load_pretrained_backbone

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)


def preprocess(image_path: str, device: torch.device) -> torch.Tensor:
    """Resize, normalize and tensorize image — identical to inference.py"""
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# ==========================================
# 🧠 Main extraction logic
# ==========================================
def extract_embeddings(cfg: dict, data_split: str = "val") -> tuple[Path, Path]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")

    # 1. Load model — identical to inference.py
    model = MobileFaceNet(embedding_dim=128).to(device)
    weights_path = cfg["paths"].get("pretrained_weights", "models/mobilefacenet.pth")

    if Path(weights_path).exists():
        load_pretrained_backbone(model, weights_path, device)
        logger.info("✅ Pre-trained MobileFaceNet loaded")
    else:
        logger.warning(f"⚠️ Weights not found at `{weights_path}`. Using random init.")

    model.eval()

    # 2. Scan dataset
    data_dir = (
        Path(cfg["paths"]["data_dir"]) / "processed" / "1000_persons" / data_split
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"❌ Data split directory not found: {data_dir}")

    # Collect all image paths with labels
    samples = []  # List of (image_path, label)
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in sorted(class_dir.glob("*.jpg")) + sorted(
            class_dir.glob("*.png")
        ):
            samples.append((img_path, label))

    logger.info(
        f"📦 Found {len(samples)} images across {len(set(s[1] for s in samples))} classes"
    )

    if len(samples) == 0:
        raise ValueError("No images found to process.")

    # 3. Extract embeddings — ONE BY ONE, identical to inference.py
    embeddings = []
    metadata = []

    logger.info("🔍 Extracting embeddings (single-image mode, like inference.py)...")
    for img_path, label in tqdm(samples, desc="Images"):
        try:
            tensor = preprocess(str(img_path), device)  # [1, 3, 112, 112]

            with torch.no_grad():
                emb = model(tensor)  # [1, 128], already L2-normed in model.forward()

            # Move to CPU and flatten — identical to inference.py
            emb_np = emb.cpu().numpy().flatten()  # [128]

            # Check norm (should be ~1.0)
            norm = np.linalg.norm(emb_np)
            if norm < 0.99 or norm > 1.01:
                logger.warning(f"⚠️ Embedding norm off for {img_path.name}: {norm:.4f}")

            embeddings.append(emb_np)
            metadata.append(
                {
                    "label": label,
                    "absolute_path": str(img_path.resolve()),
                    "relative_path": str(img_path.relative_to(data_dir)),
                }
            )

        except Exception as e:
            logger.warning(f"⚠️ Failed to process {img_path.name}: {e}")
            continue

        # Optional: clear MPS cache periodically
        if device.type == "mps" and len(embeddings) % 100 == 0:
            torch.mps.empty_cache()

    if len(embeddings) == 0:
        raise RuntimeError("❌ No embeddings extracted. Check logs for errors.")

    # 4. Stack and validate
    embeddings_array = np.stack(embeddings).astype(np.float32)  # [N, 128]

    # 🔍 Final sanity check
    norms = np.linalg.norm(embeddings_array, axis=1)
    logger.info(
        f"📊 Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
    )

    if norms.mean() < 0.95:
        logger.error(
            "❌ Embeddings are NOT L2-normalized. Something is wrong with model.forward() or preprocessing."
        )
        return None, None

    # 5. Save
    exp_dir = Path(cfg["paths"]["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    emb_path = exp_dir / "embeddings.npy"
    meta_path = exp_dir / "metadata.csv"

    np.save(emb_path, embeddings_array)

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "absolute_path", "relative_path"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    logger.info(
        f"✅ Extracted {embeddings_array.shape[0]} embeddings ({embeddings_array.shape[1]}-dim)"
    )
    logger.info(f"💾 Saved: {emb_path}")
    logger.info(f"💾 Saved: {meta_path}")

    return emb_path, meta_path


# ==========================================
# 🖥️ CLI Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract face embeddings (inference.py-style)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "full"]
    )
    args = parser.parse_args()

    try:
        cfg = load_config()
        extract_embeddings(cfg, data_split=args.split)
        logger.info("🎉 Extraction completed successfully!")
    except Exception as e:
        logger.error(f"💥 Extraction failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
