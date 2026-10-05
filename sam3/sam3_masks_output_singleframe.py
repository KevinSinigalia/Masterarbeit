# %%
# SAM3-Masken für jeden Frame einzeln (Bild-Modus, ohne Zeitkontext / Tracking).
#
# Liest die zugeschnittenen Frames aus recordings_counted/cropped_frames/frames_fishvideoXX_cropped/
# und speichert pro Frame eine komprimierte Pickle-Datei in outputs_raw_masks_compressed_singleframe/.
# Format wie bei sam3_masks_output.py, damit die Evaluation-Notebooks die Dateien direkt lesen können:
#   singleframe_results_video_XXX_NNNN_to_NNNN.pkl.gz -> {0: {"out_obj_ids", "out_binary_masks", "out_probs"}}
#
# Beispiele:
#   python sam3_masks_output_singleframe.py                      # alle Videos
#   python sam3_masks_output_singleframe.py --video_nr 30 31     # nur Video 30 und 31
#   python sam3_masks_output_singleframe.py --video_nr 30 --start 1 --end 50
import os
import sys
import argparse
import gzip
import pickle
import re
from pathlib import Path

# Argumente parsen
parser = argparse.ArgumentParser()
parser.add_argument("--video_nr", type=int, nargs="*", default=None, help="Videonummern (Standard: alle)")
parser.add_argument("--start", type=int, default=None, help="Erster Frame (inklusive)")
parser.add_argument("--end", type=int, default=None, help="Letzter Frame (inklusive)")
parser.add_argument("--prompt", type=str, default="Fish", help="Text-Prompt für SAM3")
parser.add_argument("--conf", type=float, default=0.5, help="Konfidenz-Schwelle für SAM3-Detektionen")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).parent.absolute()
FRAMES_ROOT = SCRIPT_DIR.parent / "recordings_counted" / "cropped_frames"
output_folder = SCRIPT_DIR / "outputs_raw_masks_compressed_singleframe"
output_folder.mkdir(parents=True, exist_ok=True)

# Videoordner suchen: frames_fishvideoXX_cropped
pattern = re.compile(r"^frames_fishvideo(\d+)_cropped$")
video_dirs = sorted(
    (d for d in FRAMES_ROOT.iterdir() if d.is_dir() and pattern.match(d.name)),
    key=lambda d: int(pattern.match(d.name).group(1)),
)
if args.video_nr:
    video_dirs = [d for d in video_dirs if int(pattern.match(d.name).group(1)) in args.video_nr]

# Alle zu bearbeitenden Frames sammeln (bereits vorhandene Ergebnisse überspringen)
jobs = []
for video_dir in video_dirs:
    video_nr = int(pattern.match(video_dir.name).group(1))
    for frame_path in sorted(video_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not frame_path.stem.isdigit():
            continue
        frame_nr = int(frame_path.stem)
        if args.start is not None and frame_nr < args.start:
            continue
        if args.end is not None and frame_nr > args.end:
            continue
        save_path = output_folder / f"singleframe_results_video_{video_nr:03d}_{frame_nr:04d}_to_{frame_nr:04d}.pkl.gz"
        if not save_path.exists():
            jobs.append((video_nr, frame_nr, frame_path, save_path))

print(f"{len(video_dirs)} Video(s) in {FRAMES_ROOT}, {len(jobs)} Frame(s) zu bearbeiten.")
if not jobs:
    sys.exit(0)

# %%
# Umgebungsvariable für Speicheroptimierung (muss vor torch importiert werden)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
import torch
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Modell initialisieren
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model(device=device)
processor = Sam3Processor(model, device=device, confidence_threshold=args.conf)

# %%
# Jeden Frame einzeln segmentieren
for i, (video_nr, frame_nr, frame_path, save_path) in enumerate(jobs, start=1):
    image = Image.open(frame_path).convert("RGB")

    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=args.prompt, state=state)

    # (N, 1, H, W) -> (N, H, W), in uint8 umwandeln (spart Platz)
    binary_masks = state["masks"].squeeze(1).cpu().numpy().astype(np.uint8)
    probs = state["scores"].float().cpu().numpy()

    # Gleiches Format wie sam3_masks_output.py (Schlüssel = Frame-Index relativ zum Dateistart)
    frame_output = {
        0: {
            "out_obj_ids": np.arange(len(binary_masks)),
            "out_binary_masks": binary_masks,
            "out_probs": probs,
        }
    }

    # KOMPRIMIERT SPEICHERN
    with gzip.open(save_path, "wb") as f:
        pickle.dump(frame_output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[{i}/{len(jobs)}] Video {video_nr} Frame {frame_nr:04d}: {len(binary_masks)} Masken -> {save_path.name}")

    del state
    if device == "cuda":
        torch.cuda.empty_cache()

print("--- FERTIG ---")
