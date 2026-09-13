# %%
import os
import sys
import argparse
import gzip
import pickle
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

# Argumente parsen
parser = argparse.ArgumentParser()
parser.add_argument("--video_nr", type=int, required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
args = parser.parse_args()

VIDEO_NR = args.video_nr
START_FRAME = args.start
END_FRAME = args.end

# Zielverzeichnis und Dateipfad direkt definieren
output_folder = Path("./outputs_raw_masks_compressed")
output_folder.mkdir(parents=True, exist_ok=True)

save_filename = f"tracking_results_video_{VIDEO_NR:03d}_{START_FRAME:04d}_to_{END_FRAME:04d}.pkl.gz"
save_path = output_folder / save_filename

# Prüfen, ob die Zieldatei bereits existiert -> Direkt abbrechen falls ja
if save_path.exists():
    print(f"File '{save_path}' bereits vorhanden. Überspringe Verarbeitung.")
    sys.exit(0)

# %%
# Umgebungsvariable für Speicheroptimierung (muss vor torch importiert werden)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
import sam3
import torch
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image

# ---------get the frames for sam3
FRAMES_DIR = Path(f"videos/frames_fishvideo{VIDEO_NR}")
WORKING_DIR = Path("videos")

# Frames für diesen Durchlauf filtern
video_frames_paths = list(
    filter(
        lambda frame: START_FRAME <= int(frame.stem) <= END_FRAME,
        (FRAMES_DIR.glob("*.jpg")),
    )
)

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


# %%
def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


# %%
# Modell initialisieren
torch.autocast("cuda", dtype=torch.float16).__enter__()
gpus_to_use = [0]
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

torch.cuda.empty_cache()
torch.autocast("cuda", dtype=torch.float16).__enter__()

# %%
# SAM3 Session starten
with TemporaryDirectory(dir=WORKING_DIR, prefix="tmp_sam3frames_") as tmp:
    tmp_dir = Path(tmp)
    for video_frame_path in video_frames_paths:
        shutil.copy(video_frame_path, tmp_dir / video_frame_path.name)
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(tmp_dir),
        )
    )
session_id = response["session_id"]

# Session zurücksetzen und ersten Prompt setzen
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

prompt_text_str = "Fish"
frame_idx = 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
    )
)

# Propagation starten
outputs_per_frame = propagate_in_video(predictor, session_id)

# %%
# -------- ERGEBNISSE FILTERN UND ALS DICT SPEICHERN --------
filtered_outputs = {}

for frame_idx, data in outputs_per_frame.items():
    # In uint8 umwandeln (spart Platz, ndarray wird beibehalten)
    binary_masks_uint8 = data["out_binary_masks"].astype(np.uint8)

    filtered_outputs[frame_idx] = {
        "out_obj_ids": data["out_obj_ids"],
        "out_binary_masks": binary_masks_uint8
    }

# KOMPRIMIERT SPEICHERN
print(f"Komprimiere und speichere Ergebnisse in: {save_path} ...")
with gzip.open(save_path, "wb") as f:
    # protocol=pickle.HIGHEST_PROTOCOL sorgt für maximale Geschwindigkeit/Effizienz
    pickle.dump(filtered_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"--- FERTIG ---")
print(f"Ergebnisse für Video {VIDEO_NR} (Frame {START_FRAME}-{END_FRAME}) erfolgreich komprimiert gespeichert.")