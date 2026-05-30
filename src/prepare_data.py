# src/prepare_data.py
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from PIL import Image
from loguru import logger
from tqdm import tqdm
from config import load_config


def prepare_dataset(cfg: dict):
    raw_dir = Path(cfg["paths"]["data_dir"]) / "raw"
    tmp_dir = Path(cfg["paths"]["data_dir"]) / "tmp"
    processed_dir = Path(cfg["paths"]["data_dir"]) / "processed"
    exp_dir = Path(cfg["paths"]["exp_dir"])

    # Целевые директории
    full_dir = processed_dir / "full"
    subset_dir = processed_dir / "1000_persons"
    full_dir.mkdir(parents=True, exist_ok=True)
    (subset_dir / "train").mkdir(parents=True, exist_ok=True)
    (subset_dir / "val").mkdir(parents=True, exist_ok=True)

    # 1. Загрузка датасета в streaming-режиме (экономия RAM)
    logger.info("📦 Loading dataset via HuggingFace datasets...")
    ds = load_dataset(
        "arrow", data_files=str(raw_dir / "*.arrow"), split="train", streaming=True
    )

    # 2. Потоковое сохранение во временную структуру
    tmp_dir.mkdir(parents=True, exist_ok=True)
    id_counts = defaultdict(int)
    min_per_id = cfg["dataset"].get("min_images_per_id", 3)

    logger.info("💾 Extracting images to disk...")
    for item in tqdm(ds, desc="Processing"):
        try:
            img = item["image"].convert("RGB")
            img.load()  # Проверка целостности
            pid = str(item["label"])
            pid_dir = tmp_dir / pid
            pid_dir.mkdir(exist_ok=True)
            img.save(pid_dir / f"{id_counts[pid]:04d}.jpg", format="JPEG", quality=85)
            id_counts[pid] += 1
        except Exception:
            continue

    # Фильтрация персон с недостаточным количеством изображений
    valid_ids = {pid: cnt for pid, cnt in id_counts.items() if cnt >= min_per_id}
    logger.info(f"✅ Kept {len(valid_ids)} persons ({sum(valid_ids.values())} images)")

    # 3. Копирование ВСЕХ валидных изображений в processed/full/{pid}
    logger.info("📂 Creating full dataset in processed/full/...")
    for pid in tqdm(valid_ids, desc="Copying to full"):
        src_pid_dir = tmp_dir / pid
        dst_pid_dir = full_dir / pid
        dst_pid_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_pid_dir.glob("*.jpg"):
            shutil.copy2(img_path, dst_pid_dir / img_path.name)

    # 4. Выборка N персон для быстрого экспериментирования (по умолчанию 1000)
    n_subset = cfg["dataset"].get("subset_persons", 1000)
    n_subset = min(n_subset, len(valid_ids))  # защита, если персон меньше
    selected_ids = random.sample(list(valid_ids.keys()), n_subset)
    logger.info(f"🎯 Selected {len(selected_ids)} persons for subset")

    # 5. Стратифицированное разбиение для subset (только train/val для MVP)
    ratios = cfg["dataset"].get("subset_split_ratios", [0.8, 0.2])  # train/val
    split_counts = {"train": 0, "val": 0}
    rng = np.random.RandomState(cfg["dataset"].get("seed", 42))

    logger.info("📂 Splitting subset to processed/1000_persons/...")
    for pid in tqdm(selected_ids, desc="Splitting subset"):
        files = sorted((full_dir / pid).glob("*.jpg"))
        rng.shuffle(files)
        n = len(files)
        t_end = int(n * ratios[0])

        # Train split
        train_target = subset_dir / "train" / pid
        train_target.mkdir(parents=True, exist_ok=True)
        for f in files[:t_end]:
            shutil.copy2(f, train_target / f.name)
            split_counts["train"] += 1

        # Val split
        val_target = subset_dir / "val" / pid
        val_target.mkdir(parents=True, exist_ok=True)
        for f in files[t_end:]:
            shutil.copy2(f, val_target / f.name)
            split_counts["val"] += 1

    # 6. Очистка временных файлов и сохранение статистики
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    stats = {
        "full_dataset": {
            "total_images": sum(valid_ids.values()),
            "total_persons": len(valid_ids),
            "avg_images_per_person": round(np.mean(list(valid_ids.values())), 2),
        },
        "subset_1000_persons": {
            "persons": len(selected_ids),
            "split_sizes": split_counts,
            "total_images": sum(split_counts.values()),
        },
        "config": {
            "min_images_per_id": min_per_id,
            "subset_persons": n_subset,
            "subset_split_ratios": ratios,
        }
    }
    
    with open(exp_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("📊 Stats saved to experiments/dataset_stats.json")
    logger.info(f"✨ Full dataset: {full_dir}")
    logger.info(f"✨ Subset for MVP: {subset_dir}")
    return stats


if __name__ == "__main__":
    cfg = load_config()
    prepare_dataset(cfg)