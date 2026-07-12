import os
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

from sam3.model_builder import build_sam3_multiplex_video_predictor
from huggingface_hub import login

login("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")

predictor = build_sam3_multiplex_video_predictor()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


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


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

# "video_path" needs to be either a JPEG folder or a MP4 video file
video_path = "./videos/frames"

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

# 1. Den originalen Aufruf speichern
original_init_state = predictor.model.init_state

# 2. Eine bereinigte Version der Funktion erstellen
def patched_init_state(**kwargs):
    # Wir löschen die Parameter, die das Multiplex-Modell nicht verträgt
    kwargs.pop('offload_state_to_cpu', None)
    kwargs.pop('offload_video_to_cpu', None)
    kwargs.pop('map_log_dir', None) # Oft ein weiterer Fehlerkandidat
    return original_init_state(**kwargs)

# 3. Die Funktion im Modell durch unsere bereinigte Version ersetzen
predictor.model.init_state = patched_init_state

# 4. Jetzt erst dein eigentlicher Aufruf (Zeile 86)
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
        imgsz=512,
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

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

# finally, we reformat the outputs for visualization and plot the outputs every 60 frames
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
