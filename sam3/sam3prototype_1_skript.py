import os
import sam3
import torch
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor
from sam3.model_builder import build_sam3_video_model
from sam3.visualization_utils import prepare_masks_for_visualization


bpe_file_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
video_path = "./videos/frames"
gpus_to_use = range(torch.cuda.device_count())


predictor = build_sam3_video_predictor(
    gpus_to_use = gpus_to_use,
    bpe_path=bpe_file_path,
    compile=False,
)
def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
    ):
        torch.cuda.empty_cache()
        gc.collect()
        outputs_per_frame[response["frame_index"]] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in response["outputs"].items()
        }

    return outputs_per_frame

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
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()


response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,

        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
	async_loading_frames=True,
    )
)
session_id = response["session_id"]


prompt_text_str = "fish"
frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
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
    )
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



### Part 2
import torch
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import sam3
import torch
from PIL import Image
from sam3.visualization_utils import show_box, show_mask, show_points

video_path = "./videos/frames"



if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    print("device is cuda")

import torch
from sam3.model_builder import build_sam3_video_model

bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

sam3_model2 = build_sam3_video_model(bpe_path=bpe_path, device=device)

# 2. Das gesamte Modell auf FP16 (Half) und CUDA umstellen
sam3_model2 = sam3_model2.half().cuda()

predictor2 = sam3_model2.tracker
predictor2.backbone = sam3_model2.detector.backbone

inference_state2 = predictor2.init_state(
    video_path=video_path,
    offload_video_to_cpu=True, # Video-Bilder bleiben im RAM
    offload_state_to_cpu=True   # Gedächtnis-Tokens bleiben im RAM
)

for frame in range(0, len(video_frames)):
    object_ids = outputs_per_frame1[frame]['object_ids']
    boxes = outputs_per_frame1[frame]['boxes']

    for i, object_id in enumerate(object_ids):
        orig_box = boxes[i].to(device)  # [x1, y1, x2, y2]
        norm_box = orig_box.clone().float()
        norm_box[0] /= img_w  # x1
        norm_box[2] /= img_w  # x2
        norm_box[1] /= img_h  # y1
        norm_box[3] /= img_h  # y2

        # 3. Dummy-Tensoren für Punkte und Labels auf der GPU erstellen
        # Diese überschreiben die CPU-Standardwerte der Funktion
        # Points: [Batch, Anzahl, XY] -> [1, 0, 2]
        # Labels: [Batch, Anzahl]     -> [1, 0]
        empty_points = torch.zeros((1, 0, 2), device=device, dtype=torch.float16)
        empty_labels = torch.zeros((1, 0), device=device, dtype=torch.int32)

        # 4. Der Aufruf mit allen drei Komponenten auf dem richtigen Device
        _, out_obj_ids, low_res_masks, video_res_masks = predictor2.add_new_points_or_box(
            inference_state=inference_state2,
            frame_idx=frame_idx_test,
            obj_id=int(object_ids[i]),
            box=norm_box,  # Auf CUDA
            points=empty_points,  # Auf CUDA
            labels=empty_labels,  # Auf CUDA
        )

video_segments = {}  # video_segments contains the per-frame segmentation results
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor2.propagate_in_video(inference_state2, start_frame_idx=0, max_frame_num_to_track=2, reverse=False, propagate_preflight=True):
    print("in loop")
    video_segments[frame_idx] = {
        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(outputs_per_frame1[frame_idx]['object_ids'])
    }