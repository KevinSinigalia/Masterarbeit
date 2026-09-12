import copy
import glob
import gzip
import os
import pickle
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
from tqdm import tqdm

# SAM 3 Imports
import sam3
from sam3.model_builder import build_sam3_video_model

# ==============================================================================
# 1. SETUP & DATEN LADEN
# ==============================================================================
bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
torch.autocast("cuda", dtype=torch.float16).__enter__()

print("⏳ Lade SAM 3 Modell...")
sam3_model = build_sam3_video_model(bpe_path=bpe_path)
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone



FRAMES_DIR = "./videos/frames_fishvideo2"
file_name = "tracking_results_video_002_0001_to_0040.pkl.gz"
# Eingabe-Ordner & -Datei
INPUT_DIR = "./outputs_raw_masks_compressed"
INPUTS_DIR = os.path.join(INPUT_DIR, file_name)
# Ziel-Ordner & -Pfad für neues Dict
TARGET_DIR = "./add_missing_objects"
OUTPUT_NEW_DICT_PATH = os.path.join(TARGET_DIR, file_name)

START_FRAME = 0
END_FRAME = 39



print(f"📂 Eingabe-Datei: {INPUTS_DIR}")
print(f"💾 Speicher-Zielpfad für neues Dict: {OUTPUT_NEW_DICT_PATH}")

print("⏳ Lade Videoframes und Masken-Datei...")
with gzip.open(INPUTS_DIR, "rb") as f:
    data = pickle.load(f)

image_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
frames_list = []

for path in image_paths:
    if os.path.basename(path).startswith("."):
        continue
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames_list.append(img_rgb)

frames_array = np.array(frames_list)
video_frames = frames_array
outputs_per_frame = dict(list(data.items())[START_FRAME : END_FRAME + 1])

# ALTE KLICK-EINTRÄGE BEIM START BEREINIGEN
print("🧹 Bereinige alte Klick-Einträge aus der geladenen Datei...")
for f_idx in outputs_per_frame.keys():
    if isinstance(outputs_per_frame[f_idx], dict):
        outputs_per_frame[f_idx].pop("new_points", None)
        outputs_per_frame[f_idx].pop("new_ids", None)
        outputs_per_frame[f_idx].pop("new_clicks", None)


# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================
def consolidate_non_overlapping_masks(mask_dict):
    """Löst visuelle Masken-Überschneidungen im Overlay auf."""
    if not isinstance(mask_dict, dict) or not mask_dict:
        return mask_dict

    valid_entries = [
        (k, v) for k, v in mask_dict.items() if v is not None and np.any(v)
    ]
    if len(valid_entries) <= 1:
        return mask_dict

    obj_ids = [k for k, v in valid_entries]
    masks = [v.astype(bool) for k, v in valid_entries]

    stacked = np.stack(masks, axis=0)  # Shape (N, H, W)
    overlap_map = np.sum(stacked, axis=0)

    if np.any(overlap_map > 1):
        clean_stacked = np.zeros_like(stacked, dtype=bool)
        occupied = np.zeros(stacked.shape[1:], dtype=bool)

        for i in reversed(range(len(masks))):
            clean_stacked[i] = stacked[i] & (~occupied)
            occupied |= clean_stacked[i]

        clean_dict = copy.copy(mask_dict)
        for i, oid in enumerate(obj_ids):
            clean_dict[oid] = clean_stacked[i]
        return clean_dict

    return mask_dict


def overlay_masks(
    frame, masks, alpha=0.25, draw_borders=True, show_ids=True, font_scale=0.7
):
    output = frame.copy()
    h_frame, w_frame = frame.shape[:2]
    mask_dict = {}

    if isinstance(masks, dict) and "out_obj_ids" in masks:
        obj_ids = masks["out_obj_ids"]
        mask_data = masks.get(
            "out_binary_masks",
            masks.get("out_mask_logits", masks.get("out_masks", None)),
        )

        if mask_data is not None:
            if hasattr(mask_data, "cpu"):
                mask_data = mask_data.cpu().numpy()
            if hasattr(obj_ids, "cpu"):
                obj_ids = obj_ids.cpu().numpy()

            if np.issubdtype(mask_data.dtype, np.floating):
                mask_data = mask_data > 0.0
            else:
                mask_data = mask_data.astype(bool)

            mask_data = np.squeeze(mask_data)
            if mask_data.ndim == 2:
                mask_data = np.expand_dims(mask_data, axis=0)

            for i, obj_id in enumerate(obj_ids):
                m = mask_data[i]
                if m.shape[:2] != (h_frame, w_frame):
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (w_frame, h_frame),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                mask_dict[obj_id] = m
    else:
        mask_dict = masks

    mask_dict = consolidate_non_overlapping_masks(mask_dict)

    def get_color(obj_id):
        seed_val = abs(hash(str(obj_id))) % 100000
        np.random.seed(seed_val)
        return np.random.randint(50, 255, size=3, dtype=np.uint8)

    for obj_id, mask in mask_dict.items():
        if mask is None:
            continue
        binary_mask = mask.astype(bool)
        if not np.any(binary_mask):
            continue

        color = get_color(obj_id)
        mask_uint8 = binary_mask.astype(np.uint8)

        output[binary_mask] = (
            (1 - alpha) * output[binary_mask] + alpha * color
        ).astype(np.uint8)

        if draw_borders:
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, color.tolist(), thickness=2)

        if show_ids:
            M = cv2.moments(mask_uint8)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                ys, xs = np.where(binary_mask)
                cX, cY = int(xs.mean()), int(ys.mean())

            text = f"ID: {obj_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(
                text, font, font_scale, thickness=2
            )
            text_x = cX - (text_w // 2)
            text_y = cY + (text_h // 2)

            cv2.putText(
                output,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                output,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    return output


def register_all_base_masks_for_frame(
    predictor, inference_state, frame_data, frame_idx
):
    """Übermittelt ALLE bestehenden Basis-Masken von ALLEN Objekten auf frame_idx an SAM 3 (Vollständiges Lazy Loading)."""
    added_count = 0
    if (
        "out_obj_ids" in frame_data
        and frame_data["out_obj_ids"] is not None
        and "out_binary_masks" in frame_data
        and frame_data["out_binary_masks"] is not None
    ):
        raw_ids = frame_data["out_obj_ids"]
        raw_masks = frame_data["out_binary_masks"]

        if hasattr(raw_ids, "cpu"):
            raw_ids = raw_ids.cpu().numpy()
        if hasattr(raw_masks, "cpu"):
            raw_masks = raw_masks.cpu().numpy()

        obj_ids_list = [int(x) for x in list(np.atleast_1d(raw_ids))]
        masks_arr = np.squeeze(raw_masks)
        if masks_arr.ndim == 2:
            masks_arr = np.expand_dims(masks_arr, axis=0)

        for i, obj_id in enumerate(obj_ids_list):
            if i < len(masks_arr):
                single_mask = masks_arr[i]
                if np.any(single_mask):
                    single_mask_tensor = torch.from_numpy(single_mask > 0)
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=single_mask_tensor,
                    )
                    added_count += 1
                    print(f"✔ Frame {frame_idx}: Basis-Maske für ID {obj_id} hinzugefügt.")
    return added_count


# ==============================================================================
# FENSTER 1: INITIALER VIEWER & ID-AUSWAHL UND ID-LÖSCHUNG
# ==============================================================================
class InitialViewerWindow(tk.Tk):

    def __init__(self, video_frames, outputs_per_frame, overlay_masks_func):
        super().__init__()
        self.title("Fenster 1: Masken-Vorschau & ID-Auswahl / ID Löschen (Vergleichs-Ansicht)")

        self.video_frames = video_frames
        self.outputs_per_frame = outputs_per_frame
        self.overlay_masks = overlay_masks_func

        self.H_orig, self.W_orig = self.video_frames[0].shape[:2]

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        max_w = int((screen_w * 0.88 - 30) / 2)
        max_h = int(screen_h * 0.72)
        scale = min(max_w / self.W_orig, max_h / self.H_orig)

        self.display_width = max(250, int(self.W_orig * scale))
        self.display_height = max(180, int(self.H_orig * scale))

        self.view_x0, self.view_y0 = 0, 0
        self.view_x1, self.view_y1 = self.W_orig, self.H_orig
        self.drag_start_x, self.drag_start_y = 0, 0

        self.frame_keys = (
            list(self.outputs_per_frame.keys())
            if isinstance(self.outputs_per_frame, dict)
            else list(range(len(self.outputs_per_frame)))
        )
        self.total_frames = len(self.frame_keys)
        self.current_frame_idx = 0
        self.selected_ids = []

        self._refresh_detected_ids()

        self._build_ui()
        self.update_display()

    def _refresh_detected_ids(self):
        """Sucht alle aktuell vorhandenen IDs in allen Frames."""
        self.all_detected_ids = set()
        for f_data in self.outputs_per_frame.values():
            if isinstance(f_data, dict) and "out_obj_ids" in f_data and f_data["out_obj_ids"] is not None:
                ids = f_data["out_obj_ids"]
                if hasattr(ids, "cpu"):
                    ids = ids.cpu().numpy()
                for i in np.atleast_1d(ids):
                    self.all_detected_ids.add(int(i))
        self.sorted_detected_ids = sorted(list(self.all_detected_ids))

    def _build_ui(self):
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        ttk.Button(
            controls_frame, text="◀ Zurück", command=self._on_prev
        ).pack(side=tk.LEFT, padx=5)

        self.slider = ttk.Scale(
            controls_frame,
            from_=0,
            to=self.total_frames - 1,
            orient="horizontal",
            command=self._on_slider_change,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(
            controls_frame, text="Weiter ▶", command=self._on_next
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            controls_frame, text="🔍 Reset Zoom", command=self._reset_zoom
        ).pack(side=tk.LEFT, padx=5)

        # Container für ID-Aktionen (Bearbeiten & Löschen)
        action_container = ttk.Frame(self)
        action_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        # Zeile 1: IDs zum Bearbeiten eingeben
        id_edit_frame = ttk.Frame(action_container)
        id_edit_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(
            id_edit_frame,
            text="IDs zum Bearbeiten in Fenster 2 (kommagetrennt):",
            font=("Arial", 9, "bold"),
        ).pack(side=tk.LEFT, padx=5)

        self.ids_entry = ttk.Entry(id_edit_frame, width=30)
        self.ids_entry.insert(0, ", ".join(str(x) for x in self.sorted_detected_ids))
        self.ids_entry.pack(side=tk.LEFT, padx=5)

        submit_btn = ttk.Button(
            id_edit_frame,
            text="Weiter zu Fenster 2 🚀",
            command=self._on_confirm_ids,
        )
        submit_btn.pack(side=tk.LEFT, padx=10)

        # Zeile 2: NEU - Schaltfläche zum Löschen einer ID über alle Frames
        id_delete_frame = ttk.Frame(action_container)
        id_delete_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(
            id_delete_frame,
            text="ID löschen (komplett aus allen Frames entfernen):",
            font=("Arial", 9, "bold"),
            foreground="red",
        ).pack(side=tk.LEFT, padx=5)

        self.delete_combobox = ttk.Combobox(
            id_delete_frame,
            values=[str(x) for x in self.sorted_detected_ids],
            state="readonly",
            width=10,
        )
        if self.sorted_detected_ids:
            self.delete_combobox.current(0)
        self.delete_combobox.pack(side=tk.LEFT, padx=5)

        delete_btn = ttk.Button(
            id_delete_frame,
            text="🗑️ ID löschen",
            command=self._on_delete_id,
        )
        delete_btn.pack(side=tk.LEFT, padx=5)

        self.info_label = ttk.Label(
            self,
            text="💡 Tipp: Links = Mit Masken | Rechts = Original-Frame. Mausrad = Zoom | Rechte Maustaste = Bild bewegen",
            font=("Arial", 10, "italic"),
        )
        self.info_label.pack(side=tk.TOP, pady=2)

        images_container = ttk.Frame(self)
        images_container.pack(side=tk.TOP, pady=5)

        left_sub = ttk.Frame(images_container)
        left_sub.pack(side=tk.LEFT, padx=5)
        ttk.Label(left_sub, text="Mit Masken-Overlay", font=("Arial", 10, "bold")).pack(side=tk.TOP, pady=2)
        self.image_label_left = ttk.Label(left_sub)
        self.image_label_left.pack(side=tk.TOP)

        right_sub = ttk.Frame(images_container)
        right_sub.pack(side=tk.LEFT, padx=5)
        ttk.Label(right_sub, text="Original-Frame (Ohne Masken)", font=("Arial", 10, "bold")).pack(side=tk.TOP, pady=2)
        self.image_label_right = ttk.Label(right_sub)
        self.image_label_right.pack(side=tk.TOP)

        for label in (self.image_label_left, self.image_label_right):
            label.bind("<MouseWheel>", self._on_mouse_wheel)
            label.bind("<Button-4>", self._on_mouse_wheel)
            label.bind("<Button-5>", self._on_mouse_wheel)
            label.bind("<Button-3>", self._start_pan)
            label.bind("<B3-Motion>", self._do_pan)

    def _on_delete_id(self):
        """Löscht die ausgewählte ID samt Masken vollständig aus allen Frames."""
        selected_val = self.delete_combobox.get()
        if not selected_val:
            messagebox.showwarning("Keine ID gewählt", "Bitte wähle eine ID aus der Liste aus!")
            return

        target_id = int(selected_val)

        confirm = messagebox.askyesno(
            "ID löschen bestätigen",
            f"Möchtest du ID {target_id} wirklich unwiderruflich aus allen Frames löschen?",
        )
        if not confirm:
            return

        deleted_frames_count = 0

        for f_idx, frame_data in self.outputs_per_frame.items():
            if not isinstance(frame_data, dict):
                continue

            if "out_obj_ids" in frame_data and frame_data["out_obj_ids"] is not None:
                orig_ids = frame_data["out_obj_ids"]
                orig_masks = frame_data.get("out_binary_masks", None)

                if hasattr(orig_ids, "cpu"):
                    orig_ids = orig_ids.cpu().numpy()
                if orig_masks is not None and hasattr(orig_masks, "cpu"):
                    orig_masks = orig_masks.cpu().numpy()

                ids_list = [int(x) for x in list(np.atleast_1d(orig_ids))]

                if target_id in ids_list:
                    idx = ids_list.index(target_id)
                    ids_list.pop(idx)
                    deleted_frames_count += 1

                    if orig_masks is not None:
                        masks_arr = np.squeeze(orig_masks)
                        if masks_arr.ndim == 2:
                            masks_arr = np.expand_dims(masks_arr, axis=0)
                        masks_list = [masks_arr[i] for i in range(len(masks_arr))]
                        if idx < len(masks_list):
                            masks_list.pop(idx)

                        if masks_list:
                            final_masks = np.stack(masks_list, axis=0)
                        else:
                            final_masks = np.empty((0, self.H_orig, self.W_orig), dtype=bool)
                    else:
                        final_masks = np.empty((0, self.H_orig, self.W_orig), dtype=bool)

                    frame_data["out_obj_ids"] = np.array(ids_list, dtype=np.int64)
                    frame_data["out_binary_masks"] = final_masks

        # IDs-Liste neu berechnen
        self._refresh_detected_ids()

        # Combobox und Textfeld aktualisieren
        self.delete_combobox["values"] = [str(x) for x in self.sorted_detected_ids]
        if self.sorted_detected_ids:
            self.delete_combobox.current(0)
        else:
            self.delete_combobox.set("")

        self.ids_entry.delete(0, tk.END)
        self.ids_entry.insert(0, ", ".join(str(x) for x in self.sorted_detected_ids))

        self.update_display()

        messagebox.showinfo(
            "Löschen erfolgreich",
            f"ID {target_id} wurde auf {deleted_frames_count} Frames erfolgreich entfernt!",
        )

    def _reset_zoom(self):
        self.view_x0, self.view_y0 = 0, 0
        self.view_x1, self.view_y1 = self.W_orig, self.H_orig
        self.update_display()

    def _start_pan(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _do_pan(self, event):
        dx_canvas = event.x - self.drag_start_x
        dy_canvas = event.y - self.drag_start_y

        cur_w = self.view_x1 - self.view_x0
        cur_h = self.view_y1 - self.view_y0

        dx_orig = int((dx_canvas / self.display_width) * cur_w)
        dy_orig = int((dy_canvas / self.display_height) * cur_h)

        if dx_orig == 0 and dy_orig == 0:
            return

        new_x0 = max(0, min(self.W_orig - cur_w, self.view_x0 - dx_orig))
        new_y0 = max(0, min(self.H_orig - cur_h, self.view_y0 - dy_orig))

        self.view_x0 = new_x0
        self.view_x1 = new_x0 + cur_w
        self.view_y0 = new_y0
        self.view_y1 = new_y0 + cur_h

        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_display()

    def _on_mouse_wheel(self, event):
        if event.num == 4:
            factor = 0.8
        elif event.num == 5:
            factor = 1.25
        elif hasattr(event, "delta") and event.delta != 0:
            factor = 0.8 if event.delta > 0 else 1.25
        else:
            return

        cur_w = self.view_x1 - self.view_x0
        cur_h = self.view_y1 - self.view_y0

        if factor < 1.0 and (cur_w < 30 or cur_h < 30):
            return

        focus_x = self.view_x0 + (event.x / self.display_width) * cur_w
        focus_y = self.view_y0 + (event.y / self.display_height) * cur_h

        new_w = max(30, min(self.W_orig, cur_w * factor))
        new_h = max(30, min(self.H_orig, cur_h * factor))

        rel_x = max(0.0, min(1.0, event.x / self.display_width))
        rel_y = max(0.0, min(1.0, event.y / self.display_height))

        new_x0 = max(0, focus_x - rel_x * new_w)
        new_y0 = max(0, focus_y - rel_y * new_h)

        new_x0 = max(0, min(self.W_orig - new_w, new_x0))
        new_y0 = max(0, min(self.H_orig - new_h, new_y0))

        self.view_x0 = int(new_x0)
        self.view_x1 = int(new_x0 + new_w)
        self.view_y0 = int(new_y0)
        self.view_y1 = int(new_y0 + new_h)

        self.update_display()

    def _on_prev(self):
        val = int(self.slider.get())
        if val > 0:
            self.slider.set(val - 1)

    def _on_next(self):
        val = int(self.slider.get())
        if val < self.total_frames - 1:
            self.slider.set(val + 1)

    def _on_slider_change(self, val):
        self.current_frame_idx = int(float(val))
        self.update_display()

    def _on_confirm_ids(self):
        raw_text = self.ids_entry.get().strip()

        if not raw_text:
            if self.all_detected_ids:
                self.selected_ids = sorted(list(self.all_detected_ids))
            else:
                self.selected_ids = [1]
            print(
                f"ℹ️ Feld war leer. Automatische Auswahl aller IDs: {self.selected_ids}"
            )
            self.destroy()
            return

        try:
            self.selected_ids = [
                int(x.strip()) for x in raw_text.split(",") if x.strip()
            ]
            print(f"✅ Ausgewählte IDs: {self.selected_ids}")
            self.destroy()
        except ValueError:
            messagebox.showerror(
                "Ungültige Eingabe",
                "Bitte gib die IDs als Zahlen kommagetrennt ein (z.B. 18, 19, 99)!",
            )

    def update_display(self):
        key = self.frame_keys[self.current_frame_idx]

        rendered_left = self.overlay_masks(
            self.video_frames[key],
            self.outputs_per_frame[key],
            alpha=0.25,
            draw_borders=True,
            show_ids=True,
        )

        rendered_right = self.video_frames[key].copy()

        crop_x0 = max(0, min(self.W_orig - 1, int(self.view_x0)))
        crop_x1 = max(crop_x0 + 10, min(self.W_orig, int(self.view_x1)))
        crop_y0 = max(0, min(self.H_orig - 1, int(self.view_y0)))
        crop_y1 = max(crop_y0 + 10, min(self.H_orig, int(self.view_y1)))

        cropped_left = rendered_left[crop_y0:crop_y1, crop_x0:crop_x1]
        cropped_right = rendered_right[crop_y0:crop_y1, crop_x0:crop_x1]

        if cropped_left.size == 0 or cropped_left.shape[0] == 0 or cropped_left.shape[1] == 0:
            cropped_left = rendered_left
            cropped_right = rendered_right

        resized_left = cv2.resize(
            cropped_left,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        img_tk_left = ImageTk.PhotoImage(image=Image.fromarray(resized_left))
        self.image_label_left.config(image=img_tk_left)
        self.image_label_left.image = img_tk_left

        resized_right = cv2.resize(
            cropped_right,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        img_tk_right = ImageTk.PhotoImage(image=Image.fromarray(resized_right))
        self.image_label_right.config(image=img_tk_right)
        self.image_label_right.image = img_tk_right


# ==============================================================================
# FENSTER 2: INTERAKTIVES REFINEMENT & TRACKING (SIDE-BY-SIDE)
# ==============================================================================
class InteractiveRefinementWindow(tk.Tk):

    def __init__(
        self,
        video_frames,
        video_segments,
        outputs_per_frame,
        predictor,
        inference_state,
        overlay_masks_func,
        ids_to_correct,
        output_file_path,
    ):
        super().__init__()
        self.title("Fenster 2: SAM 3 Interactive Refinement & Tracking (Vergleichs-Ansicht)")

        self.video_frames = video_frames
        self.video_segments = copy.deepcopy(video_segments)
        self.outputs_per_frame = outputs_per_frame
        self.predictor = predictor
        self.inference_state = inference_state
        self.overlay_masks = overlay_masks_func
        self.ids_to_correct = ids_to_correct
        self.output_file_path = output_file_path

        self.H_orig, self.W_orig = self.video_frames[0].shape[:2]

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        max_w = int((screen_w * 0.88 - 30) / 2)
        max_h = int(screen_h * 0.72)
        scale = min(max_w / self.W_orig, max_h / self.H_orig)

        self.display_width = max(250, int(self.W_orig * scale))
        self.display_height = max(180, int(self.H_orig * scale))

        self.view_x0, self.view_y0 = 0, 0
        self.view_x1, self.view_y1 = self.W_orig, self.H_orig
        self.drag_start_x, self.drag_start_y = 0, 0

        self.frame_keys = (
            sorted(list(self.video_segments.keys()))
            if isinstance(self.video_segments, dict)
            else list(range(len(self.video_segments)))
        )
        self.total_frames = len(self.frame_keys)
        self.current_frame_idx = 0

        self.collected_clicks = {}
        self.applied_clicks = {}
        self.new_single_frame_masks = {}

        self.registered_base_frames = set()

        self._build_ui()
        self.update_id_dropdown()
        self.update_display()

    def _build_ui(self):
        nav_frame = ttk.Frame(self)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        ttk.Button(nav_frame, text="◀ Zurück", command=self._on_prev).pack(
            side=tk.LEFT, padx=5
        )

        self.slider = ttk.Scale(
            nav_frame,
            from_=0,
            to=self.total_frames - 1,
            orient="horizontal",
            command=self._on_slider_change,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(nav_frame, text="Weiter ▶", command=self._on_next).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(
            nav_frame, text="🔍 Reset Zoom", command=self._reset_zoom
        ).pack(side=tk.LEFT, padx=5)

        refine_frame = ttk.Frame(self)
        refine_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        ttk.Label(refine_frame, text="Ziel-ID:").pack(side=tk.LEFT, padx=2)
        self.id_dropdown = ttk.Combobox(refine_frame, state="readonly", width=8)
        self.id_dropdown.pack(side=tk.LEFT, padx=5)

        self.click_type_var = tk.IntVar(value=1)
        ttk.Radiobutton(
            refine_frame,
            text="➕ Positiv",
            variable=self.click_type_var,
            value=1,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            refine_frame,
            text="➖ Negativ",
            variable=self.click_type_var,
            value=0,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            refine_frame,
            text="🗑️ Klicks löschen",
            command=self._on_reset_clicks,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            refine_frame,
            text="🔄 Update",
            command=self._on_update_points,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            refine_frame,
            text="💾 Neues Dict erstellen",
            command=self._on_create_dict,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            refine_frame,
            text="🚀 Tracking starten",
            command=self._on_start_tracking,
        ).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(
            self,
            text="Status: 0 Klicks. Klicke auf das linke oder rechte Bild -> Drücke '🔄 Update' für Vorschau.",
            font=("Arial", 10, "italic"),
        )
        self.status_label.pack(side=tk.TOP, pady=2)

        images_container = ttk.Frame(self)
        images_container.pack(side=tk.TOP, pady=5)

        left_sub = ttk.Frame(images_container)
        left_sub.pack(side=tk.LEFT, padx=5)
        ttk.Label(left_sub, text="Mit Masken-Overlay", font=("Arial", 10, "bold")).pack(side=tk.TOP, pady=2)
        self.image_label_left = ttk.Label(left_sub)
        self.image_label_left.pack(side=tk.TOP)

        right_sub = ttk.Frame(images_container)
        right_sub.pack(side=tk.LEFT, padx=5)
        ttk.Label(right_sub, text="Original-Frame (Ohne Masken)", font=("Arial", 10, "bold")).pack(side=tk.TOP, pady=2)
        self.image_label_right = ttk.Label(right_sub)
        self.image_label_right.pack(side=tk.TOP)

        for label in (self.image_label_left, self.image_label_right):
            label.bind("<Button-1>", self._handle_click)
            label.bind("<MouseWheel>", self._on_mouse_wheel)
            label.bind("<Button-4>", self._on_mouse_wheel)
            label.bind("<Button-5>", self._on_mouse_wheel)
            label.bind("<Button-3>", self._start_pan)
            label.bind("<B3-Motion>", self._do_pan)

    def _reset_zoom(self):
        self.view_x0, self.view_y0 = 0, 0
        self.view_x1, self.view_y1 = self.W_orig, self.H_orig
        self.update_display()

    def _start_pan(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _do_pan(self, event):
        dx_canvas = event.x - self.drag_start_x
        dy_canvas = event.y - self.drag_start_y

        cur_w = self.view_x1 - self.view_x0
        cur_h = self.view_y1 - self.view_y0

        dx_orig = int((dx_canvas / self.display_width) * cur_w)
        dy_orig = int((dy_canvas / self.display_height) * cur_h)

        if dx_orig == 0 and dy_orig == 0:
            return

        new_x0 = max(0, min(self.W_orig - cur_w, self.view_x0 - dx_orig))
        new_y0 = max(0, min(self.H_orig - cur_h, self.view_y0 - dy_orig))

        self.view_x0 = new_x0
        self.view_x1 = new_x0 + cur_w
        self.view_y0 = new_y0
        self.view_y1 = new_y0 + cur_h

        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_display()

    def _on_mouse_wheel(self, event):
        if event.num == 4:
            factor = 0.8
        elif event.num == 5:
            factor = 1.25
        elif hasattr(event, "delta") and event.delta != 0:
            factor = 0.8 if event.delta > 0 else 1.25
        else:
            return

        cur_w = self.view_x1 - self.view_x0
        cur_h = self.view_y1 - self.view_y0

        if factor < 1.0 and (cur_w < 30 or cur_h < 30):
            return

        focus_x = self.view_x0 + (event.x / self.display_width) * cur_w
        focus_y = self.view_y0 + (event.y / self.display_height) * cur_h

        new_w = max(30, min(self.W_orig, cur_w * factor))
        new_h = max(30, min(self.H_orig, cur_h * factor))

        rel_x = max(0.0, min(1.0, event.x / self.display_width))
        rel_y = max(0.0, min(1.0, event.y / self.display_height))

        new_x0 = max(0, focus_x - rel_x * new_w)
        new_y0 = max(0, focus_y - rel_y * new_h)

        new_x0 = max(0, min(self.W_orig - new_w, new_x0))
        new_y0 = max(0, min(self.H_orig - new_h, new_y0))

        self.view_x0 = int(new_x0)
        self.view_x1 = int(new_x0 + new_w)
        self.view_y0 = int(new_y0)
        self.view_y1 = int(new_y0 + new_h)

        self.update_display()

    def update_id_dropdown(self):
        key = self.frame_keys[self.current_frame_idx]
        frame_ids = set()
        if key in self.video_segments and isinstance(
            self.video_segments[key], dict
        ):
            if "out_obj_ids" in self.video_segments[key]:
                ids = self.video_segments[key]["out_obj_ids"]
                if hasattr(ids, "cpu"):
                    ids = ids.cpu().numpy()
                for i in ids:
                    frame_ids.add(int(i))
            else:
                frame_ids = set(self.video_segments[key].keys())

        all_available_ids = sorted(
            list(frame_ids.union(set(self.ids_to_correct)))
        )

        if all_available_ids:
            str_values = [str(i) for i in all_available_ids]
            self.id_dropdown["values"] = str_values

            current_val = self.id_dropdown.get()
            if current_val in str_values:
                self.id_dropdown.set(current_val)
            elif self.ids_to_correct and str(self.ids_to_correct[0]) in str_values:
                self.id_dropdown.set(str(self.ids_to_correct[0]))
            else:
                self.id_dropdown.current(0)

    def _on_prev(self):
        val = int(self.slider.get())
        if val > 0:
            self.slider.set(val - 1)

    def _on_next(self):
        val = int(self.slider.get())
        if val < self.total_frames - 1:
            self.slider.set(val + 1)

    def _on_slider_change(self, val):
        self.current_frame_idx = int(float(val))
        self.update_id_dropdown()
        self.update_display()

    def _handle_click(self, event):
        cur_w = self.view_x1 - self.view_x0
        cur_h = self.view_y1 - self.view_y0

        real_x = int(self.view_x0 + (event.x / self.display_width) * cur_w)
        real_y = int(self.view_y0 + (event.y / self.display_height) * cur_h)

        real_x = max(0, min(self.W_orig - 1, real_x))
        real_y = max(0, min(self.H_orig - 1, real_y))

        try:
            target_obj_id = int(self.id_dropdown.get())
        except (ValueError, TypeError):
            target_obj_id = self.ids_to_correct[0] if self.ids_to_correct else 1

        f_idx = self.current_frame_idx
        if target_obj_id not in self.collected_clicks:
            self.collected_clicks[target_obj_id] = {}

        if f_idx not in self.collected_clicks[target_obj_id]:
            self.collected_clicks[target_obj_id][f_idx] = {
                "points": [],
                "labels": [],
            }

        current_label = self.click_type_var.get()
        self.collected_clicks[target_obj_id][f_idx]["points"].append(
            [real_x, real_y]
        )
        self.collected_clicks[target_obj_id][f_idx]["labels"].append(
            current_label
        )

        total_clicks = sum(
            len(data["points"])
            for oid in self.collected_clicks
            for data in self.collected_clicks[oid].values()
        )
        self.status_label.config(
            text=f"Gesammelte Klicks: {total_clicks} Klick(s) gesetzt. Drücke '🔄 Update' für Vorschau!"
        )
        self.update_display()

    def _on_reset_clicks(self):
        self.collected_clicks = copy.deepcopy(self.applied_clicks)
        self.status_label.config(
            text="Status: Un-aktualisierte Klicks verworfen. Bestätigte Klicks & Masken bleiben erhalten!"
        )
        self.update_display()

    def _update_video_segment_mask(self, f_idx, obj_id, new_binary_mask):
        """Ersetzt/Ergänzt die Maske in self.video_segments."""
        if f_idx not in self.video_segments:
            self.video_segments[f_idx] = {
                "out_obj_ids": np.array([obj_id], dtype=np.int64),
                "out_binary_masks": np.expand_dims(new_binary_mask > 0, axis=0),
            }
            return

        frame_data = self.video_segments[f_idx]
        if isinstance(frame_data, dict) and "out_obj_ids" in frame_data:
            orig_ids = frame_data["out_obj_ids"]
            orig_masks = frame_data.get("out_binary_masks", None)

            if hasattr(orig_ids, "cpu"):
                orig_ids = orig_ids.cpu().numpy()
            if orig_masks is not None and hasattr(orig_masks, "cpu"):
                orig_masks = orig_masks.cpu().numpy()

            ids_list = [int(x) for x in list(np.atleast_1d(orig_ids))]

            if orig_masks is not None:
                masks_arr = np.squeeze(orig_masks)
                if masks_arr.ndim == 2:
                    masks_arr = np.expand_dims(masks_arr, axis=0)
                masks_list = [masks_arr[i] > 0 for i in range(len(ids_list))]
            else:
                masks_list = []

            if obj_id in ids_list:
                idx = ids_list.index(obj_id)
                masks_list[idx] = new_binary_mask > 0
            else:
                ids_list.append(obj_id)
                masks_list.append(new_binary_mask > 0)

            final_ids = np.array(ids_list, dtype=np.int64)
            if masks_list:
                final_masks = np.stack(masks_list, axis=0)
            else:
                final_masks = np.empty(
                    (0, self.H_orig, self.W_orig), dtype=bool
                )

            self.video_segments[f_idx]["out_obj_ids"] = final_ids
            self.video_segments[f_idx]["out_binary_masks"] = final_masks

    # --------------------------------------------------------------------------
    # BUTTON "UPDATE" - ERST ALLE BASIS-MASKEN DES FRAMES LADEN, DANN KLICKS ANWENDEN!
    # --------------------------------------------------------------------------
    def _on_update_points(self):
        curr_frame = self.current_frame_idx

        has_clicks_on_current_frame = any(
            curr_frame in frame_dict and frame_dict[curr_frame].get("points")
            for frame_dict in self.collected_clicks.values()
        )

        if not has_clicks_on_current_frame:
            messagebox.showwarning(
                "Keine Klicks",
                f"Bitte erst Punkte auf dem aktuellen Frame {curr_frame} setzen!",
            )
            return

        self.status_label.config(
            text=f"🔄 Lade Basis-Masken & verarbeite Klicks für Frame {curr_frame}... Bitte warten!"
        )
        self.update()

        try:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            target_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

            # 1. ERSTER EDIT AUF DIESEM FRAME -> LADE ALLE BASIS-MASKEN DES FRAMES NACH SAM 3!
            if curr_frame not in self.registered_base_frames:
                if curr_frame in self.outputs_per_frame:
                    cnt = register_all_base_masks_for_frame(
                        self.predictor,
                        self.inference_state,
                        self.outputs_per_frame[curr_frame],
                        curr_frame,
                    )
                self.registered_base_frames.add(curr_frame)

            # 2. ERST JETZT DIE KLICK-PROMPTS ANWENDEN
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device_type, dtype=target_dtype
                ):
                    for obj_id, frame_dict in self.collected_clicks.items():
                        if curr_frame in frame_dict:
                            click_data = frame_dict[curr_frame]
                            if click_data.get("points"):
                                rel_points = [
                                    [x / self.W_orig, y / self.H_orig]
                                    for x, y in click_data["points"]
                                ]

                                points_tensor = torch.tensor(
                                    rel_points, dtype=target_dtype
                                ).unsqueeze(0)
                                labels_tensor = torch.tensor(
                                    click_data["labels"], dtype=torch.int32
                                ).unsqueeze(0)

                                out_tuple = self.predictor.add_new_points_or_box(
                                    inference_state=self.inference_state,
                                    frame_idx=curr_frame,
                                    obj_id=obj_id,
                                    points=points_tensor,
                                    labels=labels_tensor,
                                    clear_old_points=True,
                                    normalize_coords=True,
                                )

                                returned_obj_ids = out_tuple[1]
                                video_res_masks = out_tuple[-1]

                                if hasattr(returned_obj_ids, "cpu"):
                                    returned_obj_ids = returned_obj_ids.cpu().numpy()
                                obj_ids_list = [int(x) for x in list(np.atleast_1d(returned_obj_ids))]

                                if obj_id in obj_ids_list:
                                    obj_batch_idx = obj_ids_list.index(obj_id)
                                    binary_mask = (video_res_masks[obj_batch_idx] > 0.0).cpu().numpy().squeeze()
                                else:
                                    binary_mask = (video_res_masks[0] > 0.0).cpu().numpy().squeeze()

                                if curr_frame not in self.new_single_frame_masks:
                                    self.new_single_frame_masks[curr_frame] = {}
                                self.new_single_frame_masks[curr_frame][
                                    obj_id
                                ] = binary_mask

                                self._update_video_segment_mask(curr_frame, obj_id, binary_mask)

            self.applied_clicks = copy.deepcopy(self.collected_clicks)

            self.status_label.config(
                text=f"✅ Klicks für Frame {curr_frame} verarbeitet & Maske aktualisiert!"
            )

        except Exception as e:
            self.status_label.config(
                text=f"❌ Fehler beim Aktualisieren: {str(e)}"
            )
            messagebox.showerror(
                "Fehler", f"Fehler beim Aktualisieren:\n{str(e)}"
            )

        self.update_display()

    # --------------------------------------------------------------------------
    # BUTTON 1: NEUES DICT ERSTELLEN (Schnelles Zusammenführen ohne Re-Inferenz)
    # --------------------------------------------------------------------------
    def _on_create_dict(self):
        target_clicks = self.collected_clicks or self.applied_clicks
        if not target_clicks and not self.new_single_frame_masks:
            messagebox.showwarning(
                "Keine Klicks",
                "Bitte erst Punkte im Bild setzen und auf '🔄 Update' drücken!",
            )
            return

        # Falls noch un-aktualisierte Klicks vorliegen, führen wir einmal Update aus
        if self.collected_clicks != self.applied_clicks:
            self._on_update_points()
            target_clicks = self.applied_clicks

        self.status_label.config(
            text="🔄 Füge korrigierte & unkorrigierte Masken zusammen... Bitte warten!"
        )
        self.update()

        try:
            new_dict = {}

            for f_idx in self.frame_keys:
                # 1. Basis-Masken und Basis-IDs für diesen Frame holen (Nicht-korrigierte Frames)
                orig_data = self.outputs_per_frame.get(f_idx, {})
                orig_ids = orig_data.get("out_obj_ids", None)
                orig_masks = orig_data.get("out_binary_masks", None)

                if orig_ids is not None:
                    if hasattr(orig_ids, "cpu"):
                        orig_ids = orig_ids.cpu().numpy()
                    ids_list = [int(x) for x in list(np.atleast_1d(orig_ids))]
                else:
                    ids_list = []

                if orig_masks is not None:
                    if hasattr(orig_masks, "cpu"):
                        orig_masks = orig_masks.cpu().numpy()
                    masks_arr = np.squeeze(orig_masks)
                    if masks_arr.ndim == 2:
                        masks_arr = np.expand_dims(masks_arr, axis=0)
                    masks_list = [masks_arr[i] > 0 for i in range(len(ids_list))]
                else:
                    masks_list = []

                new_points_info = {}

                # 2. Korrigierte Einzelbild-Masken überschreiben / ergänzen
                if f_idx in self.new_single_frame_masks:
                    for obj_id, updated_mask in self.new_single_frame_masks[f_idx].items():
                        obj_id = int(obj_id)

                        # Klick-Informationen für die Datei sichern
                        if obj_id in target_clicks and f_idx in target_clicks[obj_id]:
                            click_info = target_clicks[obj_id][f_idx]
                            new_points_info[obj_id] = {
                                "points": click_info["points"],
                                "labels": click_info["labels"],
                                "mask": updated_mask,
                            }

                        # Maske in Liste ersetzen oder als neues Objekt anhängen
                        if obj_id in ids_list:
                            idx = ids_list.index(obj_id)
                            masks_list[idx] = updated_mask > 0
                        else:
                            ids_list.append(obj_id)
                            masks_list.append(updated_mask > 0)

                # 3. Formatiere als numpy Arrays & BEREINIGE ÜBERSCHNEIDUNGEN
                frame_mask_dict = {ids_list[i]: masks_list[i] for i in range(len(ids_list))}

                cleaned_mask_dict = consolidate_non_overlapping_masks(frame_mask_dict)

                final_ids = np.array(list(cleaned_mask_dict.keys()), dtype=np.int64)
                if cleaned_mask_dict:
                    final_masks = np.stack(list(cleaned_mask_dict.values()), axis=0)
                else:
                    final_masks = np.empty((0, self.H_orig, self.W_orig), dtype=bool)

                new_dict[f_idx] = {
                    "out_obj_ids": final_ids,
                    "out_binary_masks": final_masks,
                    "new_points": new_points_info,
                }

            self.outputs_per_frame = new_dict
            self.video_segments = new_dict

            # 4. Datei als gzip/pickle speichern
            os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
            with gzip.open(self.output_file_path, "wb") as f:
                pickle.dump(new_dict, f)

            print(f"✅ Neues Dict erfolgreich gespeichert unter: {self.output_file_path}")
            self.status_label.config(
                text=f"✅ Erfolgreich! Dict gespeichert unter '{self.output_file_path}'"
            )
            messagebox.showinfo(
                "Erfolg",
                f"Neues Dict mit den aktualisierten Masken in 'out_binary_masks' wurde gespeichert unter:\n{self.output_file_path}",
            )

        except Exception as e:
            self.status_label.config(text=f"❌ Fehler: {str(e)}")
            messagebox.showerror(
                "Fehler", f"Fehler beim Erstellen des Dicts:\n{str(e)}"
            )

        self.update_display()

    # --------------------------------------------------------------------------
    # BUTTON 2: FULL TRACKING STARTEN
    # --------------------------------------------------------------------------
    def _on_start_tracking(self):
        self.status_label.config(
            text="🔄 Lade restliche Basis-Masken nach & starte Video-Tracking... Bitte Terminal beachten!"
        )
        self.update()

        try:
            if self.collected_clicks:
                self._on_update_points()

            print("⏳ Lade restliche Basis-Masken für alle Frames nach...")
            for f_idx in sorted(self.outputs_per_frame.keys()):
                if f_idx not in self.registered_base_frames:
                    register_all_base_masks_for_frame(
                        self.predictor,
                        self.inference_state,
                        self.outputs_per_frame[f_idx],
                        f_idx,
                    )
                    self.registered_base_frames.add(f_idx)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            target_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device_type, dtype=target_dtype
                ):
                    new_video_segments = {}
                    for (
                        f_idx,
                        obj_ids,
                        l_masks,
                        v_masks,
                        scores,
                    ) in tqdm(
                        self.predictor.propagate_in_video(
                            self.inference_state,
                            start_frame_idx=0,
                            max_frame_num_to_track=self.total_frames,
                            reverse=False,
                            propagate_preflight=True,
                        ),
                        total=self.total_frames,
                        desc="🚀 SAM 3 Video-Tracking",
                    ):
                        new_video_segments[f_idx] = {
                            out_obj_id: np.squeeze(
                                (v_masks[i] > 0.0).clone().cpu().numpy()
                            )
                            for i, out_obj_id in enumerate(obj_ids)
                        }

            self.video_segments = new_video_segments
            self.collected_clicks.clear()
            self.applied_clicks.clear()
            self.new_single_frame_masks.clear()
            self.status_label.config(
                text="✅ Video-Tracking erfolgreich abgeschlossen!"
            )
            messagebox.showinfo(
                "Erfolg",
                "Das Video-Tracking wurde abgeschlossen und im Fenster aktualisiert!",
            )

        except Exception as e:
            self.status_label.config(text=f"❌ Fehler beim Tracking: {str(e)}")
            messagebox.showerror("Fehler", f"Fehler beim Tracking:\n{str(e)}")

        self.update_display()

    def update_display(self):
        key = self.frame_keys[self.current_frame_idx]

        rendered_left = self.overlay_masks(
            self.video_frames[key],
            self.video_segments.get(key, {}),
            alpha=0.25,
            draw_borders=True,
            show_ids=True,
        )

        if key in self.new_single_frame_masks:
            rendered_left = self.overlay_masks(
                rendered_left,
                self.new_single_frame_masks[key],
                alpha=0.35,
                draw_borders=True,
                show_ids=True,
            )

        for obj_id, frame_dict in self.collected_clicks.items():
            if key in frame_dict:
                click_data = frame_dict[key]
                for pt, lbl in zip(click_data["points"], click_data["labels"]):
                    cx, cy = pt[0], pt[1]
                    marker_color = (0, 255, 0) if lbl == 1 else (0, 0, 255)

                    cv2.circle(
                        rendered_left,
                        (cx, cy),
                        7,
                        (0, 0, 0),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        rendered_left,
                        (cx, cy),
                        5,
                        marker_color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )

                    cv2.putText(
                        rendered_left,
                        str(obj_id),
                        (cx + 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        3,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        rendered_left,
                        str(obj_id),
                        (cx + 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )

        rendered_right = self.video_frames[key].copy()

        for obj_id, frame_dict in self.collected_clicks.items():
            if key in frame_dict:
                click_data = frame_dict[key]
                for pt, lbl in zip(click_data["points"], click_data["labels"]):
                    cx, cy = pt[0], pt[1]
                    marker_color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                    cv2.circle(rendered_right, (cx, cy), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(rendered_right, (cx, cy), 3, marker_color, -1, lineType=cv2.LINE_AA)

        crop_x0 = max(0, min(self.W_orig - 1, int(self.view_x0)))
        crop_x1 = max(crop_x0 + 10, min(self.W_orig, int(self.view_x1)))
        crop_y0 = max(0, min(self.H_orig - 1, int(self.view_y0)))
        crop_y1 = max(crop_y0 + 10, min(self.H_orig, int(self.view_y1)))

        cropped_left = rendered_left[crop_y0:crop_y1, crop_x0:crop_x1]
        cropped_right = rendered_right[crop_y0:crop_y1, crop_x0:crop_x1]

        if cropped_left.size == 0 or cropped_left.shape[0] == 0 or cropped_left.shape[1] == 0:
            cropped_left = rendered_left
            cropped_right = rendered_right

        resized_left = cv2.resize(
            cropped_left,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        img_tk_left = ImageTk.PhotoImage(image=Image.fromarray(resized_left))
        self.image_label_left.config(image=img_tk_left)
        self.image_label_left.image = img_tk_left

        resized_right = cv2.resize(
            cropped_right,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        img_tk_right = ImageTk.PhotoImage(image=Image.fromarray(resized_right))
        self.image_label_right.config(image=img_tk_right)
        self.image_label_right.image = img_tk_right


# ==============================================================================
# 3. ABLAUF-STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    print("📺 Öffne Fenster 1 (Masken-Vorschau)... Bitte IDs wählen!")
    win1 = InitialViewerWindow(
        video_frames=video_frames,
        outputs_per_frame=outputs_per_frame,
        overlay_masks_func=overlay_masks,
    )
    win1.lift()
    win1.focus_force()
    win1.mainloop()

    ids_to_correct = win1.selected_ids

    if not ids_to_correct:
        print("❌ Keine IDs ausgewählt. Skript wird beendet.")
        sys.exit()

    print(
        f"🎯 Ausgewählte IDs: {ids_to_correct}. Öffne Fenster 2..."
    )

    print("⏳ [1/2] Erstelle temporären Ordner und speichere Frames...")
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name

    for idx, frame in enumerate(frames_list[START_FRAME : END_FRAME + 1]):
        filename = f"{idx:05d}.jpg"
        filepath = os.path.join(temp_dir_path, filename)
        Image.fromarray(frame).save(filepath)

    print("⏳ [2/2] Initialisiere SAM Predictor State (Masken werden lazy geladen)...")
    inference_state = predictor.init_state(video_path=temp_dir_path)
    predictor.clear_all_points_in_video(inference_state)

    video_segments = outputs_per_frame

    print("🚀 Bereit! Öffne Fenster 2...")
    win2 = InteractiveRefinementWindow(
        video_frames=video_frames,
        video_segments=video_segments,
        outputs_per_frame=outputs_per_frame,
        predictor=predictor,
        inference_state=inference_state,
        overlay_masks_func=overlay_masks,
        ids_to_correct=ids_to_correct,
        output_file_path=OUTPUT_NEW_DICT_PATH,
    )
    win2.lift()
    win2.focus_force()
    win2.mainloop()