# src/preprocess_faces.py
import os
import cv2
import gc
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm
from config import load_config

# Глобальная переменная для рабочего процесса (модель загрузится один раз на процесс)
WORKER_APP = None


def init_worker(model_name: str, det_size: int):
    """Инициализация модели внутри дочернего процесса."""
    global WORKER_APP
    # Загружаем модель. Это займет время, но только ОДИН раз на поток.
    WORKER_APP = FaceAnalysis(
        name=model_name,
        root="./models/insightface",  # Убедись, что путь верный относительно запуска
        providers=["CPUExecutionProvider"],
    )
    # det_size=128 - баланс скорости и точности.
    # 64 слишком быстро и может пропускать лица, 128 оптимально.
    WORKER_APP.prepare(ctx_id=-1, det_size=(det_size, det_size))
    print(f"[Worker] Initialized model {model_name}")


def process_single_image(args):
    """Функция, выполняемая в каждом потоке."""
    src_path_str, dst_path_str = args
    src_path = Path(src_path_str)
    dst_path = Path(dst_path_str)

    try:
        # Чтение
        img = cv2.imread(str(src_path))
        if img is None:
            return "read_error"

        # Быстрая конвертация (если нужно)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Детекция (используем глобальную модель процесса)
        faces = WORKER_APP.get(img)

        # Освобождаем память от исходного фото сразу
        del img

        if not faces:
            return "no_face"

        # Если лиц несколько, берем самое большое
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )

        # Сохранение
        face_aligned = faces[0].normed_face
        # face_aligned уже RGB, cv2.imwrite ждет BGR
        cv2.imwrite(str(dst_path), cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR))
        del face_aligned

        return "ok"

    except Exception:
        return "error"


def preprocess_faces(cfg: dict):
    data_dir = Path(cfg["paths"]["data_dir"])
    aligned_dir = data_dir / "aligned"

    model_name = cfg.get("preprocessing", {}).get("model_name", "buffalo_s")
    det_size = cfg.get("preprocessing", {}).get("det_size", 128)

    # === КЛЮЧЕВОЙ ПАРАМЕТР ===
    # Количество параллельных потоков.
    # 4 потока = скорость x4, но и RAM x4.
    # Для MacBook с 8/16GB RAM ставим 4. Если 8GB и мало - поставь 2.
    MAX_WORKERS = 4

    print(f" Starting parallel preprocessing with {MAX_WORKERS} workers...")

    # Собираем ВСЕ задачи (пути) в один список
    tasks = []
    splits = ["train", "val", "test"]

    for split in splits:
        src_split = data_dir / "processed" / split
        dst_split = aligned_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)
        if not src_split.exists():
            continue

        for pid_dir in src_split.iterdir():
            if not pid_dir.is_dir():
                continue
            dst_pid = dst_split / pid_dir.name
            dst_pid.mkdir(exist_ok=True)

            for img_path in pid_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    dst_path = dst_pid / img_path.name
                    tasks.append((str(img_path), str(dst_path)))

    print(f"📦 Total images to process: {len(tasks)}")

    stats = {"ok": 0, "no_face": 0, "read_error": 0, "error": 0}

    # initializer=init_worker гарантирует, что модель загрузится в каждый процесс
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(model_name, det_size),
    ) as executor:
        for result in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
            if result in stats:
                stats[result] += 1

    print("\n✅ ALL DONE!")
    print(f" Stats: {stats}")
    return stats


if __name__ == "__main__":
    cfg = load_config()
    preprocess_faces(cfg)
