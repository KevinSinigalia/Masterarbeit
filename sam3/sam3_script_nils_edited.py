# %%
import os
import sys
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


# %%
VIDEO_PATH="./videos/frames"
#bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
print(sam3_root)
video_frames_paths = glob.glob(os.path.join(VIDEO_PATH, "*.jpg"))
video_frames_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
video_frames = [np.array(Image.open(f)) for f in video_frames_paths]



# %%
imgs = []
cap = cv2.VideoCapture(VIDEO_PATH)
imgs = []

# %%
def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
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
torch.autocast("cuda", dtype=torch.float16).__enter__()
gpus_to_use=[0]
#gpus_to_use = range(torch.cuda.device_count())
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use) #gpus_to_use=[0]

torch.autocast("cuda", dtype=torch.float16).__enter__()

torch.cuda.empty_cache()
torch.autocast("cuda", dtype=torch.float16).__enter__()

# Sanity check:
for name, p in predictor.model.named_parameters():
    print(name, p.dtype)
    print("-------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DEBUG OUTPUT ABOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------")
    break  # sollte torch.float16 ausgeben
# %%



# %%

video_frames_for_vis = glob.glob(os.path.join(VIDEO_PATH, "*.jpg"))
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

# %%
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=VIDEO_PATH,
    )
)
session_id = response["session_id"]

# %%
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
out = response["outputs"]
outputs_per_frame = propagate_in_video(predictor, session_id)


filtered_outputs = {}
for frame_idx, data in outputs_per_frame.items():
    filtered_outputs[frame_idx] = {
        "out_obj_ids": data["out_obj_ids"],
        "out_binary_masks": data["out_binary_masks"]
    }

# 2. Speichern des gefilterten Dictionaries
output_folder = "./outputs_raw"
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, f"tracking_results_{session_id}.pt")

torch.save(filtered_outputs, save_path)
print(f"Gefilterte Ergebnisse erfolgreich in {save_path} gespeichert.")

# finally, we reformat the outputs for visualization and plot the outputs every 60 frames
#outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

#vis_frame_stride = 50
#plt.close("all")
#for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
#    visualize_formatted_frame_output(
#        frame_idx,
#        video_frames_for_vis,
#        outputs_list=[outputs_per_frame],
#        titles=["SAM 3 Dense Tracking outputs"],
#        figsize=(6, 4),
#    )

# %%
#outputs_per_frame

# %%
#obj_ids = set()
#mask = np.zeros((len(video_frames_for_vis), video_frames_for_vis[0].shape[0], video_frames_for_vis[0].shape[1]), dtype=np.uint16)
#for frame_idx, mask_objs in outputs_per_frame.items():
#    for mask_obj_id, mask_obj in mask_objs.items():
#        mask[frame_idx, mask_obj] = mask_obj_id
#        obj_ids.add(mask_obj_id)


# 1. Ordner erstellen
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)


# 2. Durch die Frames iterieren
save_pngs = False
if save_pngs:
    print(f"Speichere Visualisierungen in {output_dir}...")
    for frame_idx in range(len(outputs_per_frame)):
        # Speicher leeren, um RAM-Probleme zu vermeiden
        plt.close('all')

        # Deine Visualisierungs-Funktion aufrufen
        # Hinweis: Diese Funktion erstellt intern ein Matplotlib-Figure-Objekt
        visualize_formatted_frame_output(
            frame_idx,
            video_frames,
            outputs_list=[prepare_masks_for_visualization({frame_idx: outputs_per_frame[frame_idx]})],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )

        # 3. Den aktuellen Plot speichern
        file_name = f"output{frame_idx + 1:03d}.png"
        file_path = os.path.join(output_dir, file_name)

        # Speichert das Bild, das gerade von visualize_... erzeugt wurde
        plt.savefig(file_path, bbox_inches='tight', dpi=150)

        if (frame_idx + 1) % 5 == 0:
            print(f"Frame {frame_idx + 1} gespeichert...")

    print(f"Fertig! Alle Bilder sind im Ordner '{output_dir}'.")

