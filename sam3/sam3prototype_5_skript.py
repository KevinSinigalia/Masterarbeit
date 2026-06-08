import torch
import numpy as np
import os
import glob
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from PIL import Image

device = Accelerator().device
model1 = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor1 = Sam3VideoProcessor.from_pretrained("facebook/sam3", dtype=torch.bfloat16,)
video_path = "./videos/frames"
video_frames_paths = glob.glob(os.path.join(video_path, "*.jpg"))
video_frames_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
video_frames = [np.array(Image.open(f)) for f in video_frames_paths]
inference_session1 = processor1.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)
# Add text prompt to detect and track objects
text = "fish"
inference_session1 = processor1.add_text_prompt(
    inference_session=inference_session1,
    text=text,
)
outputs_per_frame1 = {}
for model_outputs in model1.propagate_in_video_iterator(
    inference_session=inference_session1, max_frame_num_to_track=50
):
    processed_outputs1 = processor1.postprocess_outputs(inference_session1, model_outputs)
    outputs_per_frame1[model_outputs.frame_idx] = processed_outputs1

print(f"Processed {len(outputs_per_frame1)} frames")

from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

outputs_new_format1 = []

for frame in range(0, len(outputs_per_frame1)):
    frame_data = {}
    frame_data['out_obj_ids'] = outputs_per_frame1[frame]['object_ids'].cpu()
    frame_data['out_boxes_xywh'] = outputs_per_frame1[frame]['boxes'].cpu()
    frame_data['out_binary_masks'] = outputs_per_frame1[frame]['masks'].cpu()

    outputs_new_format1.append(frame_data)


# PART 2-----------------------------------------------------------------------
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




if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print("device is cuda")

from sam3.model_builder import build_sam3_video_model

bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

sam3_model2 = build_sam3_video_model(bpe_path=bpe_path, device=device)

sam3_model2 = sam3_model2.half().cuda()


predictor2 = sam3_model2.tracker
predictor2.backbone = sam3_model2.detector.backbone

print(f"Predictor läuft auf: {next(sam3_model2.parameters()).dtype}")

inference_state2 = predictor2.init_state(
    video_path=video_path,
    offload_video_to_cpu=True, # Video-Bilder bleiben im RAM
    offload_state_to_cpu=True   # Gedächtnis-Tokens bleiben im RAM
)

img_h, img_w = video_frames[0].shape[:2]
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
            frame_idx=frame,
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