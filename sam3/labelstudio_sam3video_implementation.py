import torch
import numpy as np
import os
import pathlib
import cv2
import tempfile
import logging
import sam3
from sam3.model_builder import build_sam3_video_model

from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from PIL import Image

#sam3 model script for the official labelstudio sam2video implementation example. Uses sam3 instead of sam2


logger = logging.getLogger(__name__)

MAX_FRAMES_TO_TRACK = 1
device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

sam3_root = "/mnt/c/Users/sinke/Desktop/Masterarbeit/sam3/sam3/sam3"
print(sam3_root)

from sam3.model_builder import build_sam3_video_predictor
gpus_to_use = range(torch.cuda.device_count())


if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone

video_path = "./videos/fishvideo1_web.mp4"
inference_state = predictor.init_state(video_path=video_path)
_last_obj_id = None
_current_model_obj_id = None



class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    def split_frames(self, video_path, temp_dir, start_frame=0, end_frame=100):
        # Open the video file
        logger.debug(f'Opening video file: {video_path}')
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # check if loaded correctly
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        else:
            # display number of frames
            logger.debug(f'Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}')
        if end_frame > total_frames:
            logger.info(f"End_frame {end_frame} ist zu hoch. Korrigiere auf {total_frames}")
            end_frame = total_frames
            
        frame_count = 0
        while True:
            # Read a frame from the video
            success, frame = video.read()
            if frame_count < start_frame:
                continue
            if frame_count + start_frame >= end_frame:
                break

            # If frame is read correctly, success is True
            if not success:
                logger.error(f'Failed to read frame {frame_count}')
                break

            # Generate a filename for the frame using the pattern with frame number: '%05d.jpg'
            frame_filename = os.path.join(temp_dir, f'{frame_count:05d}.jpg')
            if os.path.exists(frame_filename):
                logger.debug(f'Frame {frame_count}: {frame_filename} already exists')
                yield frame_filename, frame
            else:
                # Save the frame as an image file
                cv2.imwrite(frame_filename, frame)
                logger.debug(f'Frame {frame_count}: {frame_filename}')
                yield frame_filename, frame

            frame_count += 1

        # Release the video object
        video.release()

    def get_prompts(self, context) -> List[Dict]:
        if context is None or 'result' not in context:
            return []
        logger.debug(f'Extracting keypoints from context: {context}')
        prompts = []
        for ctx in context['result']:
            # Process each video tracking object separately
            obj_id = ctx['id']
            for obj in ctx['value']['sequence']:
                x = obj['x'] / 100
                y = obj['y'] / 100
                box_width = obj['width'] / 100
                box_height = obj['height'] / 100
                frame_idx = obj['frame'] - 1

                # SAM2 video works with keypoints - convert the rectangle to the set of keypoints within the rectangle

                # bbox (x, y) is top-left corner
                kps = [
                    # center of the bbox
                    [x + box_width / 2, y + box_height / 2],
                    # half of the bbox width to the left
                    [x + box_width / 4, y + box_height / 2],
                    # half of the bbox width to the right
                    [x + 3 * box_width / 4, y + box_height / 2],
                    # half of the bbox height to the top
                    [x + box_width / 2, y + box_height / 4],
                    # half of the bbox height to the bottom
                    [x + box_width / 2, y + 3 * box_height / 4]
                ]

                points = np.array(kps, dtype=np.float32)
                labels = np.array([1] * len(kps), dtype=np.int32)
                prompts.append({
                    'points': points,
                    'labels': labels,
                    'frame_idx': frame_idx,
                    'obj_id': obj_id
                })

        return prompts

    def _get_fps(self, context):
        # 1. Versuche es aus dem Kontext
        if context and 'result' in context and context['result']:
            val = context['result'][0].get('value', {})
            return val.get('framesCount'), val.get('duration')

        # 2. Falls das nicht geht, schaue ich später im predict-Teil
        # (dafür geben wir erst mal None zurück)
        return None, None

    # def convert_mask_to_bbox(self, mask):
    #     # convert mask to bbox
    #     h, w = mask.shape[-2:]
    #     mask_int = mask.reshape(h, w, 1).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     return {
    #         'x': x,
    #         'y': y,
    #         'width': w,
    #         'height': h
    #     }

    def convert_mask_to_bbox(self, mask):
        # squeeze
        mask = mask.squeeze()

        y_indices, x_indices = np.where(mask == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Find the min and max indices
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Get mask dimensions
        height, width = mask.shape

        # Calculate bounding box dimensions
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # Normalize and scale to percentage
        x_pct = (xmin / width) * 100
        y_pct = (ymin / height) * 100
        width_pct = (box_width / width) * 100
        height_pct = (box_height / height) * 100

        return {
            "x": round(x_pct, 2),
            "y": round(y_pct, 2),
            "width": round(width_pct, 2),
            "height": round(height_pct, 2)
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        global _last_obj_id  # Zugriff auf die globale Variable
        global _current_model_obj_id  # Wichtig, um die Variable zu ändern


        with torch.inference_mode():
            autocast_ctx = torch.autocast(device_type="cuda",
                                          dtype=torch.bfloat16) if device == 'cuda' else torch.inference_mode()

            with autocast_ctx:
                # 1. Prompts holen
                prompts = self.get_prompts(context)
                if not prompts and tasks and 'annotations' in tasks[0] and tasks[0]['annotations']:
                    last_anno = tasks[0]['annotations'][-1]
                    prompts = self.get_prompts({'result': last_anno['result']})

                if not prompts:
                    return ModelResponse(predictions=[])

                # --- NEU: ID-CHECK UND STATE-RESET ---
                target_id = prompts[0]['obj_id']

                if _current_model_obj_id is not None and target_id != _current_model_obj_id:
                    print(f"DEBUG: Wechsel von Objekt {_current_model_obj_id} zu {target_id}. Resette Predictor...")
                    # Das löscht alle alten Punkte und IDs aus dem Speicher des Modells
                    predictor.reset_state(inference_state)

                _current_model_obj_id = target_id
                # -------------------------------------

                # Ab hier folgt dein normaler Code...
                from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')
                task = tasks[0]
                video_url = task['data'][value]
                video_path = self.get_local_path(video_url, task_id=task['id'])

                # ID Mapping und Frame Indizes
                all_obj_ids = set(p['obj_id'] for p in prompts)
                obj_ids = {target_id: 0}
                first_frame_idx = min(p['frame_idx'] for p in prompts)
                last_frame_idx = max(p['frame_idx'] for p in prompts)

                # FPS und Metadaten
                frames_count, duration = self._get_fps(context)
                if frames_count is None and tasks and 'annotations' in tasks[0]:
                    for res in tasks[0]['annotations'][0].get('result', []):
                        if 'value' in res and 'framesCount' in res:
                            frames_count = res['value']['framesCount']
                            duration = res['value']['duration']
                            break

                fps = frames_count / duration if (frames_count and duration) else 20.0
                frames_to_track = MAX_FRAMES_TO_TRACK

                with tempfile.TemporaryDirectory() as temp_dir:
                    # Frames extrahieren
                    frames = list(self.split_frames(video_path, temp_dir,
                                                    start_frame=first_frame_idx,
                                                    end_frame=last_frame_idx + frames_to_track + 1))
                    height, width, _ = frames[0][1].shape

                    # Punkte hinzufügen
                    for prompt in prompts:
                        points_pixel = prompt['points'].copy()

                        # Wir multiplizieren NUR die Kopie mit width/height
                        points_pixel[:, 0] *= width
                        points_pixel[:, 1] *= height
                        # --------------------

                        # WICHTIG: Erstelle die Tensors aus der KOPIE (points_pixel)
                        points_t = torch.as_tensor(points_pixel, dtype=torch.float32, device=device)
                        labels_t = torch.as_tensor(prompt['labels'], dtype=torch.int32, device=device)

                        _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=prompt['frame_idx'],
                            obj_id=obj_ids[prompt['obj_id']],
                            points=prompt['points'],
                            labels=prompt['labels'],
                        )

                    # Propagation
                    sequence = []
                    for out_frame_idx, out_obj_ids, low_res, video_res, obj_scores in predictor.propagate_in_video(
                            inference_state=inference_state,
                            start_frame_idx=last_frame_idx,
                            max_frame_num_to_track=frames_to_track,
                            reverse=False,
                            propagate_preflight=True
                    ):
                        real_frame_idx = out_frame_idx + first_frame_idx
                        for i, out_obj_id in enumerate(out_obj_ids):
                            mask = (video_res[i] > 0.0).cpu().numpy()
                            bbox = self.convert_mask_to_bbox(mask)
                            if bbox:
                                sequence.append({
                                    'frame': int(real_frame_idx + 1),
                                    'x': float(bbox['x']),
                                    'y': float(bbox['y']),
                                    'width': float(bbox['width']),
                                    'height': float(bbox['height']),
                                    'enabled': True,
                                    'rotation': 0,
                                    'time': float(real_frame_idx / fps)
                                })

                    # Sequenzen zusammenbauen
                    context_result_sequence = []
                    target_id = prompts[0]['obj_id']

                    if context and 'result' in context and len(context['result']) > 0:
                        context_result_sequence = context['result'][0]['value'].get('sequence', [])
                    elif tasks and 'annotations' in tasks[0]:
                        for res in tasks[0]['annotations'][0].get('result', []):
                            if res.get('id') == target_id:
                                context_result_sequence = res['value'].get('sequence', [])
                                break

                    # WICHTIG: Nutze hier 'clean_old_sequence' (Alte Frames vor Korrektur beibehalten)
                    clean_old_sequence = [s for s in context_result_sequence if s['frame'] < (first_frame_idx + 1)]

                    prediction_result = {
                        'value': {
                            'framesCount': int(frames_count or 100),
                            'duration': float(duration or 4.0),
                            'sequence': clean_old_sequence + sequence,
                        },
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'videorectangle',
                        'origin': 'manual',
                        'id': target_id
                    }

                    return ModelResponse(predictions=[{'result': [prediction_result]}])