import os
import sam3
import torch
import gc

from sam3.model_builder import build_sam3_multiplex_video_predictor
from huggingface_hub import login

login("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")

predictor = build_sam3_multiplex_video_predictor(
    bpe_path="./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
)


video_path = "./videos/frames"

import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)



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

# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()


response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# note: in case you already ran one text prompt and now want to switch to another text prompt
# it's required to reset the session first (otherwise the results would be wrong)
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

def disable_fa3(module):
    for m in module.modules():
        if hasattr(m, "use_fa3"):
            m.use_fa3 = False

disable_fa3(predictor.model)


prompt_text_str = "fish"
frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
        use_fa3=False,
    )
)




out = response["outputs"]



gc.collect()
torch.cuda.empty_cache()

# 1. Ordner erstellen
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. Daten vorbereiten (wie gehabt)
print("Starte Propagation...")
with torch.inference_mode():
    outputs_per_frame = propagate_in_video(
        predictor,
        session_id
    )# Hinweis: Wir brauchen hier prepare_masks_for_visualization evtl. nicht mal,
# wenn wir direkt auf die Rohdaten zugreifen, aber wir behalten es für die Struktur bei.
formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)

# 3. Farben für die Masken definieren (BGR Format für OpenCV)
# Du kannst hier weitere Farben hinzufügen, falls du viele Objekte hast
COLORS = [
    (255, 0, 0),  # Blau
    (0, 255, 0),  # Grün
    (0, 0, 255),  # Rot
    (0, 255, 255),  # Gelb
    (255, 0, 255),  # Magenta
    (255, 255, 0)  # Cyan
]

print(f"Speichere Frames in '{output_dir}'...")

# 4. Alle Frames durchlaufen
# Wir sortieren die Indizes, damit die Reihenfolge stimmt
frame_indices = sorted(formatted_outputs.keys())

for i, frame_idx in enumerate(tqdm(frame_indices, desc="Saving Frames")):
    # Pfad zum Originalbild holen (da video_frames_for_vis bei dir Pfade sind)
    img_path = video_frames_for_vis[frame_idx]

    # Bild laden
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    # Masken für diesen Frame abrufen
    # Die Struktur ist meist: {obj_id: mask_array}
    masks_dict = formatted_outputs[frame_idx]

    # Jede Maske auf das Bild zeichnen
    for obj_id, mask in masks_dict.items():
        # Maske in Boolean umwandeln (falls nötig)
        mask = mask.astype(bool)

        # Farbe wählen (obj_id bestimmt die Farbe)
        color = COLORS[obj_id % len(COLORS)]

        # Eine farbige Ebene erstellen
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask] = color

        # Transparenz hinzufügen (Alpha-Blending)
        # 0.5 ist die Stärke der Maske (50% durchsichtig)
        cv2.addWeighted(colored_mask, 0.5, frame, 1.0, 0, frame)

    # Dateiname generieren (output0001.jpg, output0002.jpg, ...)
    file_name = os.path.join(output_dir, f"output{i + 1:04d}.jpg")

    # Speichern
    cv2.imwrite(file_name, frame)

print(f"\nFertig! {len(frame_indices)} Bilder wurden im Ordner '{output_dir}' gespeichert.")