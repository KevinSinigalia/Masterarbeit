import os
import sys
import torch
import numpy as np
from uuid import uuid4
from PIL import Image
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from PIL import Image

#sam3 model script for the official labelstudio sam2image implementation example. Uses sam3 instead of sam2

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush

# Den Pfad zu deinem SAM3-Ordner definieren
# Basierend auf deinem Log liegt der Hauptordner hier:
sam3_root = "/mnt/c/Users/sinke/Desktop/Masterarbeit/sam3/sam3/sam3"

if sam3_root not in sys.path:
    sys.path.append(sam3_root)
    # Falls der Ordner verschachtelt ist (sam3/sam3), fügen wir beide hinzu:
    sys.path.append(os.path.join(sam3_root, "sam3"))

print(f"DEBUG: Python Path erweitert um {sam3_root}")


# --- ABSTRAKTE BASIS (Das "Template") ---
class BaseModelWrapper(ABC):
    @abstractmethod
    def load(self):
        """Modell laden"""
        pass

    @abstractmethod
    def set_image(self, image: np.ndarray):
        """Bild-Features berechnen"""
        pass

    @abstractmethod
    def predict(self,
                point_coords: Optional[np.ndarray],
                point_labels: Optional[np.ndarray],
                input_box: Optional[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """Inferenz ausführen"""
        pass


# --- DEINE SAM 2 IMPLEMENTIERUNG ---
class SAM2Wrapper(BaseModelWrapper):
    def __init__(self):
        self.model = None
        self.predictor = None
        self.device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
        self.checkpoint = os.path.join(os.getcwd(), "checkpoints",
                                       os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt'))

    def load(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self.device == 'cuda':
            # Optimierung für moderne GPUs
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


        self.model = build_sam2(self.config, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print(f"SAM 2.1 erfolgreich geladen auf {self.device}")

    def set_image(self, image: np.ndarray):
        pil_img = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
        self.width, self.height = pil_img.size

        # Wir nutzen inference_mode und autocast nur lokal für diese Operation
        with torch.inference_mode():
            if self.device == 'cuda':
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    self.inference_state = self.processor.set_image(pil_img)
            else:
                self.inference_state = self.processor.set_image(pil_img)

    def predict(self, point_coords, point_labels, input_box):
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        # Wir nehmen die Maske mit dem höchsten Score
        idx = np.argsort(scores)[::-1]
        best_mask = masks[idx[0]].astype(np.uint8)
        best_score = float(scores[idx[0]])

        return [best_mask], [best_score]


class SAM3Wrapper(BaseModelWrapper):
    def __init__(self):
        self.model = None
        self.processor = None
        self.inference_state = None
        self.device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Pfade basierend auf deinem Beispiel
        self.bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        # Falls du einen spezifischen Checkpoint nutzt, hier anpassen:

    def load(self):
        """Lädt das SAM 3 Modell nach deinem Beispiel"""
        try:
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model.box_ops import box_xywh_to_cxcywh
            from sam3.visualization_utils import normalize_bbox
            self.box_xywh_to_cxcywh = box_xywh_to_cxcywh
            self.normalize_bbox = normalize_bbox

        except ImportError as e:
            print(f"Fehler beim Importieren von SAM3: {e}")
            raise
        if self.device == 'cuda':
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        self.model = build_sam3_image_model(bpe_path=self.bpe_path, device=self.device)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        print("SAM 3 Modell und Processor erfolgreich geladen.")

    def set_image(self, image: np.ndarray):
        """Berechnet den Inference State für das Bild"""
        # wie im example notebook
        pil_img = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
        self.width, self.height = pil_img.size
        self.inference_state = self.processor.set_image(pil_img)

    def predict(self, point_coords, point_labels, input_box):
        """Vorhersage mit SAM 3 Logik - Extraktion aus dem inference_state"""
        if self.inference_state is None:
            return [], []

        with torch.inference_mode():
            autocast_ctx = torch.autocast(device_type="cuda",
                                          dtype=torch.bfloat16) if self.device == 'cuda' else torch.inference_mode()

            with autocast_ctx:
                self.processor.reset_all_prompts(self.inference_state)
                #wie in example notebook
                if input_box is not None:
                    x, y, x2, y2 = input_box
                    w, h = x2 - x, y2 - y
                    box_xywh = torch.tensor([float(x), float(y), float(w), float(h)]).view(-1, 4)
                    box_cxcywh = self.box_xywh_to_cxcywh(box_xywh)
                    norm_box = self.normalize_bbox(box_cxcywh, self.width, self.height).flatten().tolist()

                    self.inference_state = self.processor.add_geometric_prompt(
                        state=self.inference_state,
                        box=norm_box,
                        label=True
                    )

                # 3. Punkt-Prompts (falls vorhanden)
                if point_coords is not None and len(point_coords) > 0:
                    for coord, label in zip(point_coords, point_labels):
                        norm_point = [coord[0] / self.width, coord[1] / self.height]
                        self.inference_state = self.processor.add_geometric_prompt(
                            state=self.inference_state,
                            point=norm_point,
                            label=bool(label)
                        )

                masks = self.inference_state.get("masks")
                scores = self.inference_state.get("iou_predictions") or self.inference_state.get("scores")

        # Sicherheitcheck: Haben wir Daten erhalten?
        if masks is None or len(masks) == 0:
            print("Warnung: Keine Masken im inference_state gefunden.")
            return [], []

        # Falls masks ein Torch-Tensor ist, zu numpy konvertieren
        if hasattr(masks, 'cpu'):
            # Erst auf CPU schieben, dann zu Float32 konvertieren, dann zu Numpy
            masks = masks.detach().cpu().float().numpy()

        if hasattr(scores, 'cpu'):
            # Erst auf CPU schieben, dann zu Float32 konvertieren, dann zu Numpy
            scores = scores.detach().cpu().float().numpy()
            # ------------------------

            # Die Masken sind oft im Format (N, 1, H, W) oder (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

            # Beste Maske basierend auf dem Score auswählen
            # Wir nutzen flatten(), um sicherzugehen, dass wir ein 1D-Array für argsort haben
        idx = np.argsort(scores.flatten())[::-1]
        best_mask = masks[idx[0]].astype(np.uint8)
        best_score = float(scores.flatten()[idx[0]])

        return [best_mask], [best_score]


# --- LABEL STUDIO ML BACKEND KLASSE ---
# Der Name MUSS "NewModel" sein, damit _wsgi.py den Import findet
class NewModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(NewModel, self).__init__(**kwargs)

        # Hier wird SAM 2 initialisiert.
        # Willst du später SAM 3, tauschst du einfach diesen Wrapper aus.
        self.wrapper = SAM3Wrapper()
        #self.wrapper.load()

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Wird aufgerufen, wenn in Label Studio interagiert wird."""
        if self.wrapper.model is None:
            self.wrapper.load()
            
        if not context or not context.get('result'):
            return ModelResponse(predictions=[])

        # 1. Metadaten aus dem XML-Interface holen
        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        # 2. Bild laden und vorbereiten
        img_url = tasks[0]['data'][value]
        image_path = self.get_local_path(img_url, task_id=tasks[0].get('id'))
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        width, height = img_pil.size

        # 3. Interaktionen (Punkte/Boxen) sammeln
        point_coords, point_labels, input_box, selected_label = [], [], None, None

        for ctx in context['result']:
            # Umrechnung von % in Pixel
            x = ctx['value']['x'] * width / 100
            y = ctx['value']['y'] * height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]

            if ctx_type == 'keypointlabels':
                # Falls 'is_positive' nicht existiert, nehmen wir 1 (positiver Klick)
                is_pos = ctx.get('is_positive', True)
                point_labels.append(1 if is_pos else 0)
                point_coords.append([int(x), int(y)])

            elif ctx_type == 'rectanglelabels':
                bw = ctx['value']['width'] * width / 100
                bh = ctx['value']['height'] * height / 100
                input_box = [int(x), int(y), int(x + bw), int(y + bh)]

        # 4. Modell-Inferenz via Wrapper
        self.wrapper.set_image(img_np)
        masks, probs = self.wrapper.predict(
            point_coords=np.array(point_coords) if point_coords else None,
            point_labels=np.array(point_labels) if point_labels else None,
            input_box=np.array(input_box) if input_box is not None else None
        )

        # 5. Ergebnisse in Label Studio Brush-Format (RLE) umwandeln
        final_results = []
        for mask, prob in zip(masks, probs):
            # mask * 255 macht aus 0/1 Werten 0/255 für das Brush-Tool
            rle = brush.mask2rle(mask * 255)
            final_results.append({
                'id': str(uuid4())[:4],
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [selected_label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return ModelResponse(predictions=[{
            'result': final_results,
            'score': np.mean(probs) if probs else 0
        }])