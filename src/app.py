# src/app.py
"""
Gradio UI for Face Recognition MVP.
Handles uploads safely, saves to disk, and searches via FAISS.
"""

import sys
import os
import random
import numpy as np
import torch
import gradio as gr
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
from loguru import logger

# 🔒 1. Fix paths BEFORE any other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 🔒 2. Reproducibility & macOS stability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# 🔒 3. Safe imports
try:
    from src.config import load_config
    from src.search import FaceSearcher
except ImportError as e:
    logger.error(f"🚨 Import failed: {e}")
    logger.info("💡 Run: pip install faiss-cpu gradio loguru")
    sys.exit(1)

# Upload directory
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 🧠 Core Functions
# ==========================================
def get_example_paths(data_root: str, n: int = 3) -> list:
    """Dynamically find real dataset images for Gradio examples."""
    samples = []
    root = Path(data_root) / "processed" / "1000_persons" / "val"
    if not root.exists():
        return []
    for class_dir in sorted(root.iterdir()):
        if len(samples) >= n or not class_dir.is_dir():
            continue
        for p in class_dir.glob("*.jpg"):
            samples.append(str(p))
            if len(samples) >= n:
                break
    return samples


def save_uploaded_image(image_input) -> Path | None:
    """Saves Gradio upload (numpy/PIL) to disk and returns path."""
    if image_input is None:
        return None
    if isinstance(image_input, (str, Path)):
        return Path(image_input) if Path(image_input).exists() else None

    try:
        img = (
            PILImage.fromarray(image_input)
            if isinstance(image_input, np.ndarray)
            else image_input
        )
        img = img.convert("RGB")
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        save_path = UPLOAD_DIR / filename
        img.save(save_path, "JPEG", quality=95)
        return save_path
    except Exception as e:
        logger.error(f"❌ Failed to save upload: {e}")
        return None


_searcher = None


def get_searcher() -> FaceSearcher:
    global _searcher
    if _searcher is None:
        cfg = load_config()
        _searcher = FaceSearcher(cfg)
        logger.info("✅ FaceSearcher initialized")
    return _searcher


def recognize_face(
    image_input, top_k: int = 5, threshold: float = 0.35
) -> tuple[list, str]:
    image_path = save_uploaded_image(image_input)
    if not image_path:
        return [], "⚠️ Не удалось сохранить изображение"

    try:
        matches = get_searcher().search(
            str(image_path), k=int(top_k), threshold=float(threshold)
        )
        if not matches:
            return [], f"❌ Нет совпадений > {threshold:.2f}"

        # 🔧 FIX: Преобразуем относительные пути в абсолютные
        cfg = load_config()
        base_dirs = [
            Path(cfg["paths"]["data_dir"]) / "processed" / "1000_persons" / "val",
            Path(cfg["paths"]["data_dir"]) / "processed" / "1000_persons" / "train",
            Path(cfg["paths"]["data_dir"]) / "processed" / "full",
        ]

        results = []
        for m in matches:
            abs_path = None
            for base in base_dirs:
                candidate = base / m["path"]
                if candidate.exists():
                    abs_path = candidate
                    break  # Нашли файл → выходим из цикла поиска

            if abs_path:
                results.append(
                    (str(abs_path), f"👤 {m['label']} | 🔗 {m['similarity']:.3f}")
                )

        return results, f"✅ Найдено: {len(results)} совпадений"

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return [], f"💥 Ошибка: {str(e)[:100]}"


# ==========================================
# 🎨 Gradio UI
# ==========================================
def create_demo():
    cfg = load_config()
    examples = [[p] for p in get_example_paths(cfg["paths"]["data_dir"])]

    custom_css = """
    .gallery-container { max-height: 500px; overflow-y: auto; border-radius: 8px; }
    .header { text-align: center; margin-bottom: 1rem; }
    .gradio-container { max-width: 950px; margin: auto; }
    """

    with gr.Blocks(
        title="🕵️ Face Recognition MVP",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            "# 🕵️ Face Recognition MVP\n### Загрузите фото → найдите совпадения\n*MobileFaceNet • FAISS • Apple MPS*",
            elem_classes="header",
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="📷 Фото лица", type="numpy", height=280)
                with gr.Accordion("⚙️ Настройки", open=False):
                    slider_k = gr.Slider(1, 15, 5, step=1, label="Топ-K")
                    slider_thr = gr.Slider(
                        0.0, 1.0, 0.35, step=0.05, label="Порог уверенности"
                    )
                btn_search = gr.Button("🔍 Найти", variant="primary")
                btn_clear = gr.Button("🗑️ Сбросить", variant="secondary")

            with gr.Column(scale=2):
                gallery_out = gr.Gallery(
                    label="✅ Совпадения",
                    columns=3,
                    height=380,
                    object_fit="cover",
                    elem_classes="gallery-container",
                )
                status_out = gr.Textbox(
                    label="📊 Статус", value="🟢 Готово к поиску", interactive=False
                )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=img_input,
                outputs=[gallery_out, status_out],
                fn=lambda p: recognize_face(p, 5, 0.35),
                label="📌 Примеры из датасета",
            )

        btn_search.click(
            fn=recognize_face,
            inputs=[img_input, slider_k, slider_thr],
            outputs=[gallery_out, status_out],
        )
        btn_clear.click(
            fn=lambda: ([], "🟢 Готово к поиску"),
            inputs=None,
            outputs=[gallery_out, status_out],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    logger.info(f"🎨 Запуск на http://localhost:{args.port}")
    create_demo().launch(
        server_port=args.port, share=args.share, server_name="127.0.0.1"
    )
