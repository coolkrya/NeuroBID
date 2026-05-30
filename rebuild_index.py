# rebuild_index.py (положите в корень проекта)
"""
🧹 Clean & rebuild face embeddings + FAISS index.
Usage: python rebuild_index.py [--split train|val|full] [--batch-size 32]
"""

import subprocess
import sys
from pathlib import Path
import argparse
from loguru import logger

# Resolve project root for relative imports & paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import load_config
except ImportError:
    logger.error("❌ Cannot load config. Ensure you're running from project root.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Rebuild embeddings & FAISS index")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "full"]
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for MPS/CPU"
    )
    args = parser.parse_args()

    cfg = load_config()
    exp_dir = Path(cfg["paths"]["exp_dir"])

    # 1. Clean old artifacts
    files_to_remove = ["embeddings.npy", "metadata.csv", "index.faiss"]
    logger.info(f"🧹 Cleaning old artifacts in `{exp_dir}`...")
    for f in files_to_remove:
        p = exp_dir / f
        if p.exists():
            p.unlink()
            logger.info(f"   🗑️ Removed {f}")
        else:
            logger.debug(f"   ⏭️ Skipped {f} (not found)")

    # 2. Run extraction
    logger.info(
        f"🚀 Starting extract_embeddings.py (split={args.split}, bs={args.batch_size})"
    )
    cmd = [sys.executable, "src/extract_embeddings.py", "--split", args.split]

    # Set OpenMP env vars for macOS stability
    env = {**__import__("os").environ}
    if sys.platform == "darwin":
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        logger.info("✅ Index rebuilt successfully! Ready for `search.py` or `app.py`.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Extraction failed with code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
