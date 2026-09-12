import os
import re
import gzip
import pickle
import random
import shutil
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ================= KONFIGURATION =================
MASKS_DIR = Path("./outputs_raw_masks_compressed")
VIDEOS_DIR = Path("./videos")
OUTPUT_DIR = Path("./dataset")

MAX_FILES = None  # Auf None setzen für alle
VAL_SPLIT = 0.1  # 10% der Chunks gehen ins Validierungsset
CLASS_ID = 0  # 0 für 'fish'
CLASS_NAME = "fish"
RANDOM_SEED = 42

# WICHTIG: Begrenzt auf 4 Prozesse, um den RAM nicht zu überlasten!
# Falls du 32GB/64GB RAM hast, kannst du vorsichtig auf 6 oder 8 erhöhen.
NUM_WORKERS = min(4, os.cpu_count() or 1)


# =================================================


def mask_to_yolo_polygon(mask, img_w, img_h, epsilon_factor=0.002):
    """Wandelt eine Binärmaske in normalisierte YOLO-Polygon-Koordinaten um."""
    mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        norm_coords = []
        for point in approx.reshape(-1, 2):
            x = np.clip(point[0] / img_w, 0.0, 1.0)
            y = np.clip(point[1] / img_h, 0.0, 1.0)
            norm_coords.extend([f"{x:.6f}", f"{y:.6f}"])

        polygons.append(norm_coords)

    return polygons


def process_single_pickle(args):
    """Verarbeitet genau eine Pickle-Datei isoliert und gibt Speicher sofort frei."""
    pkl_file, split, pattern_str, videos_dir_str, output_dir_str, class_id = args

    videos_dir = Path(videos_dir_str)
    output_dir = Path(output_dir_str)
    pattern = re.compile(pattern_str)

    match = pattern.search(pkl_file.name)
    if not match:
        match = re.search(r"(\d+)_(\d+)_to_(\d+)", pkl_file.name)
        if not match:
            return f"Fehler: Konnte {pkl_file.name} nicht parsen"

    video_nr_int = int(match.group(1))
    start_frame_int = int(match.group(2))
    end_frame_int = int(match.group(3))

    chunk_folder_name = f"video_{video_nr_int:03d}_{start_frame_int:04d}_to_{end_frame_int:04d}"

    target_img_dir = output_dir / "images" / split / chunk_folder_name
    target_lbl_dir = output_dir / "labels" / split / chunk_folder_name
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    src_frames_dir = videos_dir / f"frames_fishvideo{video_nr_int}"
    if not src_frames_dir.exists():
        return f"Fehler: Bilderordner {src_frames_dir} existiert nicht"

    # Pickle laden
    data = None
    try:
        with gzip.open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return f"Fehler beim Laden von {pkl_file.name}: {e}"

    # Frames abarbeiten
    try:
        for frame_key, frame_dict in data.items():
            if isinstance(frame_key, int):
                abs_frame_nr = start_frame_int + frame_key if frame_key < start_frame_int else frame_key
            else:
                abs_frame_nr = int(frame_key)

            img_name = f"{abs_frame_nr:04d}.jpg"
            src_img_path = src_frames_dir / img_name

            if not src_img_path.exists():
                continue

            img = cv2.imread(str(src_img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            masks = frame_dict.get("out_binary_masks", [])
            label_lines = []

            for mask in masks:
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()

                polygons = mask_to_yolo_polygon(mask, w, h)
                for poly in polygons:
                    label_lines.append(f"{class_id} " + " ".join(poly))

            frame_base_name = f"{abs_frame_nr:04d}"

            # Dateioperationen
            shutil.copy(src_img_path, target_img_dir / f"{frame_base_name}.jpg")

            dst_lbl = target_lbl_dir / f"{frame_base_name}.txt"
            with open(dst_lbl, 'w') as f:
                for line in label_lines:
                    f.write(line + "\n")

    finally:
        # Explizite Freigabe des RAMs für diesen Prozess
        del data
        gc.collect()

    return None


def main():
    random.seed(RANDOM_SEED)

    pickle_files = sorted(list(MASKS_DIR.glob("*.pkl")) + list(MASKS_DIR.glob("*.pkl.gz")))
    if not pickle_files:
        print(f"[!] Keine Dateien im Ordner {MASKS_DIR} gefunden.")
        return

    if MAX_FILES is not None:
        pickle_files = pickle_files[:MAX_FILES]
        print(f"[TESTMODUS] Es werden die ersten {len(pickle_files)} Pickle-Dateien verarbeitet.\n")
    else:
        print(f"Gefundene Pickle-Dateien: {len(pickle_files)}")

    shuffled_files = list(pickle_files)
    random.shuffle(shuffled_files)

    if len(shuffled_files) == 1:
        val_files = set()
    else:
        val_count = max(1, int(len(shuffled_files) * VAL_SPLIT))
        val_files = set(shuffled_files[:val_count])

    pattern_str = r"tracking_res[u|y]?lk?ts_video_(\d+)_(\d+)_to_(\d+)\.pkl"

    tasks = []
    for pkl_file in pickle_files:
        split = "val" if pkl_file in val_files else "train"
        tasks.append((
            pkl_file,
            split,
            pattern_str,
            str(VIDEOS_DIR),
            str(OUTPUT_DIR),
            CLASS_ID
        ))

    print(f"Starte Parallelisierung mit {NUM_WORKERS} Prozessen (RAM-schonend)...")

    # max_tasks_per_child=1 startet nach jedem Job einen frischen Worker-Prozess,
    # damit sich keinerlei RAM-Lecks über die Zeit aufstauen können!
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, max_tasks_per_child=1) as executor:
        futures = [executor.submit(process_single_pickle, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Verarbeite Chunks parallel"):
            err = future.result()
            if err:
                print(f"\n[!] {err}")

    # data.yaml erstellen
    yaml_content = {
        'path': str(OUTPUT_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            CLASS_ID: CLASS_NAME
        }
    }

    yaml_file = OUTPUT_DIR / "data.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n[+] Abgeschlossen! Datensatz liegt in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()