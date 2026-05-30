import sys, os
import torch, cv2, faiss, numpy as np
from loguru import logger
from src.config import load_config

logger.info(f"Python {sys.version}")
logger.info(
    f"PyTorch {torch.__version__} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
)
logger.info(f"OpenCV {cv2.__version__}")
logger.info(f"FAISS {faiss.__version__}")
logger.info(f"NumPy {np.__version__}")

cfg = load_config()
logger.info(f"✅ Config loaded. Dataset: {cfg['dataset']['name']}")
logger.info("🚀 Окружение готово к работе!")
