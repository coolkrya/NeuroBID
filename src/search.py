"""
Face Search Module using FAISS for fast similarity search.
Requires: pip install faiss-cpu
"""

import torch
import numpy as np
import csv
from pathlib import Path
from PIL import Image
from torchvision import transforms
from loguru import logger
from typing import List, Dict, Optional

import os
import sys

from config import load_config
from model import MobileFaceNet, load_pretrained_backbone

# 1. Удаляем стандартный обработчик (по умолчанию он выводит время)
logger.remove()

# 2. Добавляем новый с нужным форматом (без {time})
logger.add(sys.stderr, format="{level} | {name}:{line} - {message}")

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)


if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault(
        "OMP_NUM_THREADS", "1"
    )  # Avoid thread oversubscription on MPS


try:
    import faiss
except ImportError:
    logger.error("❌ FAISS not installed. Run: pip install faiss-cpu")
    raise


class FaceSearcher:
    """
    High-level API for face identification.
    Loads pre-trained model, FAISS index, and metadata.
    Provides fast top-K search with cosine similarity thresholding.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or load_config()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.exp_dir = Path(self.cfg["paths"]["exp_dir"])
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load model
        self.model = self._load_model()
        self.model.eval()

        # 2. Load or build FAISS index + metadata
        self.index, self.metadata = self._load_or_build_index()
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError(
                "❌ FAISS index is empty. Run src/extract_embeddings.py first."
            )

        # 3. Preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        logger.info(
            f"✅ FaceSearcher ready | Index: {self.index.ntotal} vectors | "
            f"Device: {self.device} | Dim: {self.index.d}"
        )

    def _load_model(self) -> MobileFaceNet:
        model = MobileFaceNet(embedding_dim=128).to(self.device)
        weights_path = self.cfg["paths"].get(
            "pretrained_weights", "models/mobilefacenet.pth"
        )

        if Path(weights_path).exists():
            load_pretrained_backbone(model, weights_path, self.device)
        else:
            logger.warning(
                f"⚠️ Pretrained weights not found at `{weights_path}`. Using random init."
            )
        return model

    def _load_or_build_index(self):
        index_path = self.exp_dir / "index.faiss"
        meta_path = self.exp_dir / "metadata.csv"

        # Try loading cached index
        """
        if index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            metadata = self._load_metadata(meta_path)
            logger.info(f"📥 Loaded cached FAISS index & metadata")
            return index, metadata
        """

        # Fallback: build from embeddings.npy
        emb_path = self.exp_dir / "embeddings.npy"
        if not emb_path.exists():
            logger.error(
                "❌ No index or embeddings.npy found. Run extract_embeddings.py first."
            )
            return None, []

        logger.info("🔨 Building FAISS index from embeddings.npy...")
        embeddings = np.load(emb_path).astype(np.float32)

        # 🔍 DEBUG: Проверка норм векторов
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(
            f"📊 Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
        )

        if norms.mean() < 0.99 or norms.mean() > 1.01:
            logger.warning(
                "⚠️ Embeddings not L2-normalized. Normalizing now for FAISS..."
            )
            embeddings = embeddings / (norms[:, np.newaxis] + 1e-9)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, str(index_path))
        logger.info(f"💾 Index saved: {index_path}")

        metadata = self._load_metadata(meta_path) if meta_path.exists() else []
        return index, metadata

    def _load_metadata(self, path: Path) -> List[Dict[str, str]]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _preprocess(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def search(
        self, image_path: str, k: int = 5, threshold: float = 0.35
    ) -> List[Dict]:
        """
        Search for top-K similar faces.

        Args:
            image_path: Path to query image
            k: Number of results to return
            threshold: Min cosine similarity to include (0.0-1.0)

        Returns:
            List of dicts: [{'similarity', 'label', 'path', 'index'}, ...]
        """
        # 1. Extract embedding
        try:
            tensor = self._preprocess(image_path)
            embedding = self.model(tensor).cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"❌ Failed to process query image: {e}")
            return []

        # 2. FAISS search (returns similarities, not distances)
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(embedding, k)

        # 3. Format & filter results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS padding
                continue
            if sim < threshold:
                continue

            meta = self.metadata[idx]
            results.append(
                {
                    "similarity": round(float(sim), 4),
                    "label": meta.get("label", "unknown"),
                    "path": meta.get("relative_path", meta.get("filepath", "unknown")),
                    "index": int(idx),
                }
            )

        # Already sorted by FAISS, but explicit sort for safety
        return sorted(results, key=lambda x: x["similarity"], reverse=True)


# ==========================================
# 🖥️ CLI Testing
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Search CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to query image")
    parser.add_argument("--k", type=int, default=5, help="Top-K results")
    parser.add_argument("--threshold", type=float, default=0.35, help="Min similarity")
    args = parser.parse_args()

    logger.info(f"🔍 Searching for: {args.image}")
    searcher = FaceSearcher()
    matches = searcher.search(args.image, k=args.k, threshold=args.threshold)

    if not matches:
        logger.warning("⚠️ No matches found above threshold.")
    else:
        logger.info("✅ Top matches:")
        for i, m in enumerate(matches, 1):
            logger.info(
                f"  {i}. Label: {m['label']} | Sim: {m['similarity']:.4f} "  # | Path: {m['path']}
            )
