import glob
import gzip
import os
import pickle
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# SAM 3 Imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
import sam3
from sam3.model_builder import build_sam3_video_model

# ==============================================================================
# 1. SETUP & PFADE CONFIG
# ==============================================================================
bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

FRAMES_DIR = "./videos/frames_fishvideo1"
INPUT_FILE = (
    "./add_missing_objects/tracking_results_video_001_0001_to_0040_refined.pkl.gz"
)
OUTPUT_FILE = (
    "./add_missing_objects/tracking_results_video_001_0001_to_0040_refined.pkl.gz"
)

START_FRAME = 0
END_FRAME = 39

device_type = "cuda" if torch.cuda.is_available() else "cpu"
target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ==============================================================================
# 2. MODELL & DATEN LADEN
# ==============================================================================
print("⏳ Lade SAM 3 Modell...")
torch.autocast(device_type, dtype=target_dtype).__enter__()
sam3_model = build_sam3_video_model(bpe_path=bpe_path)
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone

print(f"📂 Eingabe-Datei: {INPUT_FILE}")
print(f"💾 Ziel-Datei:   {OUTPUT_FILE}")

print("⏳ Lade Videoframes und Masken-Datei...")
with gzip.open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

image_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
frames_list = []

for path in image_paths:
    if os.path.basename(path).startswith("."):
        continue
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames_list.append(img_rgb)

outputs_per_frame = dict(list(data.items())[START_FRAME : END_FRAME + 1])
sub_frames_list = frames_list[START_FRAME : END_FRAME + 1]

H_orig, W_orig = sub_frames_list[0].shape[:2]


# ==============================================================================
# 3. HELFER: LETZTE KLICK-FRAMES ERMITTELN & MASKEN REGISTRIEREN
# ==============================================================================
def get_last_click_frame_per_obj(outputs_per_frame):
    """Ermittelt pro Object-ID den höchsten Frame-Index, an dem ein Klick existiert."""
    last_click_frames = {}
    for f_idx, f_data in outputs_per_frame.items():
        new_points_dict = f_data.get("new_points", {})
        for obj_id, click_data in new_points_dict.items():
            points = click_data.get("points", [])
            if len(points) > 0:
                obj_id = int(obj_id)
                last_click_frames[obj_id] = max(
                    last_click_frames.get(obj_id, -1), f_idx
                )
    return last_click_frames


def register_masks_to_predictor(
    predictor, inference_state, outputs_per_frame, last_click_frames
):
    """Übermittelt Basis-Masken NUR BIS zum Frame des letzten Klicks pro Objekt."""
    masks_added_count = 0
    masks_skipped_count = 0

    for frame_idx in sorted(outputs_per_frame.keys()):
        frame_data = outputs_per_frame[frame_idx]

        if "out_obj_ids" not in frame_data or frame_data["out_obj_ids"] is None:
            continue

        raw_ids = frame_data["out_obj_ids"]
        raw_masks = frame_data["out_binary_masks"]

        if hasattr(raw_ids, "cpu"):
            raw_ids = raw_ids.cpu().numpy()
        if hasattr(raw_masks, "cpu"):
            raw_masks = raw_masks.cpu().numpy()

        obj_ids_list = [int(x) for x in raw_ids]
        masks_arr = np.squeeze(raw_masks)
        if masks_arr.ndim == 2:
            masks_arr = np.expand_dims(masks_arr, axis=0)

        for idx, obj_id in enumerate(obj_ids_list):
            # Prüfen, wann der letzte Klick für diese ID war
            # (Falls für die ID nie geklickt wurde, nehmen wir alle Frames)
            max_allowed_frame = last_click_frames.get(obj_id, END_FRAME)

            if frame_idx <= max_allowed_frame:
                single_mask = masks_arr[idx]
                single_mask_tensor = torch.from_numpy(single_mask)

                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=single_mask_tensor,
                )
                masks_added_count += 1
            else:
                masks_skipped_count += 1

    print(
        f"✔ {masks_added_count} Basis-Masken übergeben (bis inkl. letztem Klick-Frame)."
    )
    if masks_skipped_count > 0:
        print(
            f"ℹ️ {masks_skipped_count} spätere Basis-Masken ignoriert (werden durch Tracking neu berechnet)."
        )


# ==============================================================================
# 4. SAM 3 INITIALISIERUNG
# ==============================================================================
# 1. Höchste Frame-Indizes für Klicks analysieren
last_click_frames = get_last_click_frame_per_obj(outputs_per_frame)
for obj_id, last_f in last_click_frames.items():
    print(f"📌 Objekt-ID {obj_id}: Letzter Klick befindet sich auf Frame {last_f}")

print("⏳ [1/2] Erstelle temporären Ordner und speichere Frames...")
with tempfile.TemporaryDirectory() as temp_dir_path:
    for idx, frame in enumerate(sub_frames_list):
        filename = f"{idx:05d}.jpg"
        filepath = os.path.join(temp_dir_path, filename)
        Image.fromarray(frame).save(filepath)

    print("⏳ [2/2] Initialisiere SAM Predictor State...")
    inference_state = predictor.init_state(video_path=temp_dir_path)
    predictor.clear_all_points_in_video(inference_state)

    # 2. Basis-Masken registrieren (NUR bis inkl. letztem Klick-Frame!)
    register_masks_to_predictor(
        predictor, inference_state, outputs_per_frame, last_click_frames
    )

    # ==========================================================================
    # 5. AUTOMATISCHES EINLESEN DER "new_points"
    # ==========================================================================
    print("🎯 Lese 'new_points' aus und übergebe Punkte an SAM 3...")
    points_added_count = 0

    with torch.inference_mode():
        with torch.autocast(device_type=device_type, dtype=target_dtype):
            for frame_idx in sorted(outputs_per_frame.keys()):
                frame_data = outputs_per_frame[frame_idx]
                new_points_dict = frame_data.get("new_points", {})

                for obj_id, click_data in new_points_dict.items():
                    points = click_data.get("points", [])
                    labels = click_data.get("labels", [])

                    if not points:
                        continue

                    # Koordination auf [0.0, 1.0] normalisieren
                    rel_points = [
                        [x / W_orig, y / H_orig] for x, y in points
                    ]

                    points_tensor = torch.tensor(
                        rel_points, dtype=target_dtype
                    ).unsqueeze(0)
                    labels_tensor = torch.tensor(
                        labels, dtype=torch.int32
                    ).unsqueeze(0)

                    # Klicks/Punkte in den SAM Predictor einspeisen
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=int(obj_id),
                        points=points_tensor,
                        labels=labels_tensor,
                        clear_old_points=True,
                        normalize_coords=True,
                    )
                    points_added_count += len(points)
                    print(
                        f"  ➜ Frame {frame_idx}: {len(points)} Punkt(e) für ID {obj_id} hinzugefügt."
                    )

    # ==========================================================================
    # 6. VIDEO TRACKING PROPAGATION
    # ==========================================================================
    print("🚀 Starte SAM 3 Video-Tracking (Propagation über alle Frames)...")
    final_outputs = {}
    total_frames = len(outputs_per_frame)

    with torch.inference_mode():
        with torch.autocast(device_type=device_type, dtype=target_dtype):
            for f_idx, obj_ids, l_masks, v_masks, scores in tqdm(
                predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=total_frames,
                    reverse=False,
                    propagate_preflight=True,
                ),
                total=total_frames,
                desc="Tracking",
            ):
                # Binarisiere Masken (> 0.0 Logits)
                binary_masks = (v_masks > 0.0).cpu().numpy().astype(bool)

                # Speichere die Tracking-Ergebnisse
                final_outputs[f_idx] = {
                    "out_obj_ids": np.array(obj_ids, dtype=np.int32),
                    "out_binary_masks": binary_masks,
                    "new_points": outputs_per_frame[f_idx].get(
                        "new_points", {}
                    ),
                }

    # ==========================================================================
    # 7. ERGEBNIS SPEICHERN
    # ==========================================================================
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"💾 Speichere aktualisiertes Dictionary unter: {OUTPUT_FILE}")

    with gzip.open(OUTPUT_FILE, "wb") as f:
        pickle.dump(final_outputs, f)

    print("✅ Fertig! Das Video-Tracking wurde erfolgreich abgeschlossen.")