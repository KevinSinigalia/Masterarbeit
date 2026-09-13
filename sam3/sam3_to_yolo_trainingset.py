import os
import re
import gzip
import pickle
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
TEST_VIDEOS_TXT = VIDEOS_DIR / "test_videos.txt"

MAX_FILES = None  # Auf None setzen für alle
CLASS_ID = 0  # 0 für 'fish'
CLASS_NAME = "fish"

# Fallback-Liste der Test/Val-Videos, falls die txt-Datei nicht existiert
FALLBACK_VAL_VIDEOS = [
    "fishvideo30", "fishvideo31", "fishvideo32", "fishvideo33", "fishvideo34",
    "fishvideo38", "fishvideo39", "fishvideo40", "fishvideo56", "fishvideo57",
    "fishvideo58", "fishvideo59", "fishvideo60", "fishvideo61", "fishvideo62",
    "fishvideo63", "fishvideo64", "fishvideo65", "fishvideo66", "fishvideo67",
    "fishvideo68"
]

NUM_WORKERS = min(4, os.cpu_count() or 1)


# =================================================


def load_val_video_ids():
    """Liest die Video-IDs für das Validierungsset ein und gibt ein Set von Integern zurück."""
    val_ids = set()

    if TEST_VIDEOS_TXT.exists():
        print(f"[+] Lade Validierungs-Videos aus: {TEST_VIDEOS_TXT}")
        with open(TEST_VIDEOS_TXT, "r") as f:
            lines = f.read().split()
            for entry in lines:
                match = re.search(r"\d+", entry)
                if match:
                    val_ids.add(int(match.group(0)))
    else:
        print(f"[!] {TEST_VIDEOS_TXT} nicht gefunden. Nutze Fallback-Liste...")
        for entry in FALLBACK_VAL_VIDEOS:
            match = re.search(r"\d+", entry)
            if match:
                val_ids.add(int(match.group(0)))

    return val_ids


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
    """Verarbeitet genau eine Pickle-Datei isoliert."""
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

    data = None
    try:
        with gzip.open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return f"Fehler beim Laden von {pkl_file.name}: {e}"

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

            shutil.copy(src_img_path, target_img_dir / f"{frame_base_name}.jpg")

            dst_lbl = target_lbl_dir / f"{frame_base_name}.txt"
            with open(dst_lbl, 'w') as f:
                for line in label_lines:
                    f.write(line + "\n")

    finally:
        del data
        gc.collect()

    return None


def main():
    # 1. Val-IDs laden
    val_video_ids = load_val_video_ids()
    print(f"Festgelegte Val/Test-Videos ({len(val_video_ids)} Stück): {sorted(list(val_video_ids))}\n")

    # 2. Pickle-Dateien suchen
    pickle_files = sorted(list(MASKS_DIR.glob("*.pkl")) + list(MASKS_DIR.glob("*.pkl.gz")))
    if not pickle_files:
        print(f"[!] Keine Dateien im Ordner {MASKS_DIR} gefunden.")
        return

    if MAX_FILES is not None:
        pickle_files = pickle_files[:MAX_FILES]
        print(f"[TESTMODUS] Es werden die ersten {len(pickle_files)} Pickle-Dateien verarbeitet.\n")
    else:
        print(f"Gefundene Pickle-Dateien insgesamt: {len(pickle_files)}")

    pattern_str = r"tracking_res[u|y]?lk?ts_video_(\d+)_(\d+)_to_(\d+)\.pkl"
    pattern = re.compile(pattern_str)

    # 3. Tasks vorbereiten und explizit nach Video-ID aufteilen
    tasks = []
    train_count = 0
    val_count = 0

    for pkl_file in pickle_files:
        match = pattern.search(pkl_file.name)
        if not match:
            match = re.search(r"(\d+)_(\d+)_to_(\d+)", pkl_file.name)

        if match:
            vid_id = int(match.group(1))
            # Wenn die Video-ID in der Test-Liste steht -> 'val', sonst -> 'train'
            if vid_id in val_video_ids:
                split = "val"
                val_count += 1
            else:
                split = "train"
                train_count += 1
        else:
            split = "train"
            train_count += 1

        tasks.append((
            pkl_file,
            split,
            pattern_str,
            str(VIDEOS_DIR),
            str(OUTPUT_DIR),
            CLASS_ID
        ))

    print(f"Aufteilung der Chunks: {train_count} Train-Dateien | {val_count} Val-Dateien")
    print(f"Starte Parallelisierung mit {NUM_WORKERS} Prozessen...")

    # 4. Multiprocessing Pool ausführen
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, max_tasks_per_child=1) as executor:
        futures = [executor.submit(process_single_pickle, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Verarbeite Chunks"):
            err = future.result()
            if err:
                print(f"\n[!] {err}")

    # 5. data.yaml erstellen (portabler relativer Pfad)
    yaml_content = {
        'path': './dataset',
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
    print(f"[+] data.yaml geschrieben: {yaml_file}")


if __name__ == "__main__":
    main()