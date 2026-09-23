from collections import defaultdict
import glob
import os
import re
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageTk

# Standard-Pfade
GT_DIR = "./gt_labels_new_tofill_temp"
RECORDINGS_DIR = "./recordings_counted"

COLOR_MAP = {
    "fisch": (0, 255, 0),  # Grün
    "fishneighbour": (255, 140, 0),  # Orange
    "snail": (255, 255, 0),  # Gelb
    "reflection": (255, 0, 255),  # Magenta
}


class BoundingBoxCropAnnotatorGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("TIDE Crop-Box Annotator (Drag & Drop Rechteck)")
        self.root.geometry("1500x880")

        # Datenstrukturen
        self.current_folder_name = None
        self.video_clean_name = None
        self.task_groups = defaultdict(list)
        self.task_to_frame_map = {}

        # Aktuelle Auswahl
        self.current_task = None
        self.current_frame_nr = None

        self.orig_frame_img = None
        self.orig_mask_img = None
        self.tk_frame = None
        self.tk_mask = None

        self.scale = 1.0

        # Drag & Drop Variablen
        self.is_drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

        # Gespeicherte Box in Originalauflösung: (x1, y1, x2, y2)
        self.crop_box = None

        self._setup_ui()
        self.load_base_folders()

    def _setup_ui(self):
        # 1. Linkes Steuerungs-Panel
        left_panel = tk.Frame(self.root, width=320, padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            left_panel,
            text="1. Video-Ordner (GT):",
            font=("Arial", 10, "bold"),
        ).pack(anchor=tk.W)
        self.folder_combobox = ttk.Combobox(left_panel, state="readonly")
        self.folder_combobox.pack(fill=tk.X, pady=(2, 10))
        self.folder_combobox.bind("<<ComboboxSelected>>", self.on_folder_select)

        tk.Label(
            left_panel,
            text="2. Task ➔ Frame auswählen:",
            font=("Arial", 10, "bold"),
        ).pack(anchor=tk.W)
        self.task_listbox = tk.Listbox(
            left_panel, exportselection=False, font=("Consolas", 10)
        )
        self.task_listbox.pack(fill=tk.BOTH, expand=True, pady=(2, 10))
        self.task_listbox.bind("<<ListboxSelect>>", self.on_task_select)

        # Farb-Legende
        leg_frame = tk.LabelFrame(
            left_panel, text="GT Masken Farben", padx=5, pady=5
        )
        leg_frame.pack(fill=tk.X, pady=5)
        for name, col in COLOR_MAP.items():
            hex_c = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
            tk.Label(
                leg_frame,
                text=f"■ {name.capitalize()}",
                fg=hex_c,
                font=("Arial", 9, "bold"),
            ).pack(anchor=tk.W)

        # Status Label
        self.status_label = tk.Label(
            left_panel,
            text="Rechteck mit Maus ziehen",
            fg="blue",
            font=("Arial", 10, "bold"),
            justify=tk.LEFT,
        )
        self.status_label.pack(pady=5, fill=tk.X)

        self.btn_reset = tk.Button(
            left_panel,
            text="🔄 Box zurücksetzen",
            command=self.reset_box,
            state=tk.DISABLED,
        )
        self.btn_reset.pack(fill=tk.X, pady=2)

        self.btn_save = tk.Button(
            left_panel,
            text="💾 Crop-Box speichern (.txt)",
            command=self.save_box,
            bg="#90ee90",
            font=("Arial", 10, "bold"),
            state=tk.DISABLED,
        )
        self.btn_save.pack(fill=tk.X, pady=5)

        # 2. Rechtes Anzeigefeld (Dual View)
        views_panel = tk.Frame(self.root, bg="#1a1a1a")
        views_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Frame Canvas (Links)
        frame_box = tk.Frame(views_panel, bg="#222")
        frame_box.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2), pady=2
        )
        self.lbl_frame_title = tk.Label(
            frame_box,
            text="🎥 ORIGINAL FRAME (Ziehen zum Schneiden)",
            fg="white",
            bg="#333",
            font=("Arial", 10, "bold"),
        )
        self.lbl_frame_title.pack(fill=tk.X)

        self.canvas_frame = tk.Canvas(frame_box, bg="#111", cursor="cross")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self._bind_mouse_events(self.canvas_frame)

        # Mask Canvas (Rechts)
        mask_box = tk.Frame(views_panel, bg="#222")
        mask_box.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0), pady=2
        )
        self.lbl_mask_title = tk.Label(
            mask_box,
            text="🎭 GT MASKE (Referenz)",
            fg="white",
            bg="#333",
            font=("Arial", 10, "bold"),
        )
        self.lbl_mask_title.pack(fill=tk.X)

        self.canvas_mask = tk.Canvas(mask_box, bg="#111", cursor="cross")
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)
        self._bind_mouse_events(
            self.canvas_mask
        )  # Auch hier kann gezogen werden!

    def _bind_mouse_events(self, canvas):
        canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        canvas.bind("<B1-Motion>", self.on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def load_base_folders(self):
        abs_gt = os.path.abspath(GT_DIR)
        if not os.path.exists(abs_gt):
            messagebox.showerror(
                "Fehler", f"Ordner '{GT_DIR}' nicht gefunden!"
            )
            return

        folders = [
            f
            for f in sorted(os.listdir(abs_gt))
            if os.path.isdir(os.path.join(abs_gt, f))
        ]
        if folders:
            self.folder_combobox["values"] = folders
            self.folder_combobox.current(0)
            self.on_folder_select(None)

    def load_mapping_file(self, gt_folder_path, video_name):
        txt_candidates = [
            os.path.join(gt_folder_path, f"{video_name}.txt"),
            os.path.join(gt_folder_path, f"{self.current_folder_name}.txt"),
            os.path.join(GT_DIR, f"{video_name}.txt"),
        ]
        for t in glob.glob(os.path.join(gt_folder_path, "*.txt")):
            if not t.endswith("_cut.txt") and t not in txt_candidates:
                txt_candidates.append(t)

        for cand in txt_candidates:
            if os.path.exists(cand):
                with open(cand, "r") as f:
                    return [int(x) for x in re.findall(r"\d+", f.read())]
        return []

    def on_folder_select(self, event):
        self.current_folder_name = self.folder_combobox.get()
        gt_folder_path = os.path.abspath(
            os.path.join(GT_DIR, self.current_folder_name)
        )
        self.video_clean_name = self.current_folder_name.replace(
            "_finished", ""
        )

        self.task_groups.clear()
        self.task_to_frame_map.clear()
        self.task_listbox.delete(0, tk.END)

        npy_files = glob.glob(os.path.join(gt_folder_path, "*.npy"))
        for f in npy_files:
            match = re.search(r"task-(\d+)", os.path.basename(f))
            if match:
                self.task_groups[f"task-{match.group(1)}"].append(f)

        sorted_tasks = sorted(
            self.task_groups.keys(),
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        frame_numbers = self.load_mapping_file(
            gt_folder_path, self.video_clean_name
        )

        for i, task_key in enumerate(sorted_tasks):
            frame_nr = frame_numbers[i] if i < len(frame_numbers) else None
            self.task_to_frame_map[task_key] = frame_nr
            mask_count = len(self.task_groups[task_key])
            frame_str = (
                f"Frame {frame_nr:04d}"
                if frame_nr is not None
                else "Frame ???"
            )
            self.task_listbox.insert(
                tk.END, f"{task_key} ➔ {frame_str} ({mask_count} Masken)"
            )

        if sorted_tasks:
            self.task_listbox.selection_set(0)
            self.on_task_select(None)

    def find_frame_image(self, frame_nr):
        if frame_nr is None:
            return None
        dirs = [
            os.path.abspath(
                os.path.join(RECORDINGS_DIR, f"frames_{self.video_clean_name}")
            ),
            os.path.abspath(
                os.path.join(RECORDINGS_DIR, self.video_clean_name)
            ),
            os.path.abspath(
                os.path.join(
                    RECORDINGS_DIR, f"frames_{self.current_folder_name}"
                )
            ),
        ]
        candidates = [
            f"{frame_nr:04d}.jpg",
            f"{frame_nr:04d}.png",
            f"{frame_nr:05d}.jpg",
            f"{frame_nr:05d}.png",
            f"{frame_nr}.jpg",
            f"{frame_nr}.png",
        ]
        for d in dirs:
            if not os.path.exists(d):
                continue
            for c in candidates:
                p = os.path.join(d, c)
                if os.path.exists(p):
                    return p
        return None

    def build_composite_mask(self, npy_file_list):
        if not npy_file_list:
            return None
        first = np.load(npy_file_list[0])
        h, w = first.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for path in npy_file_list:
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            mask_bool = arr > 0
            fname = os.path.basename(path).lower()
            color = (200, 200, 200)
            for k, col in COLOR_MAP.items():
                if k in fname:
                    color = col
                    break
            rgb[mask_bool] = color
        return Image.fromarray(rgb)

    def on_task_select(self, event):
        selection = self.task_listbox.curselection()
        if not selection:
            return

        selected_text = self.task_listbox.get(selection[0])
        task_key = selected_text.split(" ")[0]
        self.current_task = task_key
        self.current_frame_nr = self.task_to_frame_map.get(task_key)

        # 1. Maske
        file_list = self.task_groups.get(task_key, [])
        self.orig_mask_img = self.build_composite_mask(file_list)

        # 2. Frame
        frame_path = self.find_frame_image(self.current_frame_nr)
        if frame_path and os.path.exists(frame_path):
            self.orig_frame_img = Image.open(frame_path)
            self.lbl_frame_title.config(
                text=f"🎥 FRAME {self.current_frame_nr:04d} ({os.path.basename(frame_path)})"
            )
        else:
            self.orig_frame_img = None
            self.lbl_frame_title.config(
                text=f"❌ FRAME {self.current_frame_nr} NICHT GEFUNDEN"
            )

        self.lbl_mask_title.config(
            text=f"🎭 GT MASKE ({task_key} mit {len(file_list)} Masken)"
        )

        self.reset_box()
        self.render_views()

    def render_views(self):
        ref_img = self.orig_frame_img or self.orig_mask_img
        if ref_img is None:
            return

        self.root.update_idletasks()
        cw = self.canvas_frame.winfo_width()
        ch = self.canvas_frame.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 500, 600

        fw, fh = ref_img.size
        self.scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * self.scale), int(fh * self.scale)

        if self.orig_frame_img:
            disp_frame = self.orig_frame_img.resize(
                (nw, nh), Image.Resampling.BILINEAR
            )
            self.tk_frame = ImageTk.PhotoImage(disp_frame)
        else:
            self.tk_frame = None

        if self.orig_mask_img:
            disp_mask = self.orig_mask_img.resize(
                (nw, nh), Image.Resampling.NEAREST
            )
            self.tk_mask = ImageTk.PhotoImage(disp_mask)
        else:
            self.tk_mask = None

        self.redraw()

    # --- MAUS-EVENTS FÜR DAS RECHTECK (DRAG & DROP) ---

    def on_mouse_press(self, event):
        ref_img = self.orig_frame_img or self.orig_mask_img
        if ref_img is None:
            return
        self.is_drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_x = event.x
        self.current_y = event.y

    def on_mouse_drag(self, event):
        if not self.is_drawing:
            return
        self.current_x = event.x
        self.current_y = event.y
        self.redraw()

    def on_mouse_release(self, event):
        if not self.is_drawing:
            return
        self.is_drawing = False
        self.current_x = event.x
        self.current_y = event.y

        ref_img = self.orig_frame_img or self.orig_mask_img
        if ref_img is None:
            return

        # Canvas-Koordinaten -> Originalpixel umrechnen
        ox1 = int(min(self.start_x, self.current_x) / self.scale)
        oy1 = int(min(self.start_y, self.current_y) / self.scale)
        ox2 = int(max(self.start_x, self.current_x) / self.scale)
        oy2 = int(max(self.start_y, self.current_y) / self.scale)

        # Clamping auf Bildgrenzen
        img_w, img_h = ref_img.size
        ox1 = max(0, min(img_w, ox1))
        oy1 = max(0, min(img_h, oy1))
        ox2 = max(0, min(img_w, ox2))
        oy2 = max(0, min(img_h, oy2))

        # Prüfen ob Box groß genug ist (mindestens 10x10 Pixel)
        if (ox2 - ox1) > 10 and (oy2 - oy1) > 10:
            self.crop_box = (ox1, oy1, ox2, oy2)
            w = ox2 - ox1
            h = oy2 - oy1
            self.status_label.config(
                text=f"Box: [{ox1}, {oy1}, {ox2}, {oy2}]\nGröße: {w} x {h} px"
            )
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)
        else:
            self.reset_box()

        self.redraw()

    def redraw(self):
        # Links: Frame zeichnen
        self.canvas_frame.delete("all")
        if self.tk_frame:
            self.canvas_frame.create_image(
                0, 0, anchor=tk.NW, image=self.tk_frame
            )
        else:
            self.canvas_frame.create_text(
                20,
                30,
                anchor=tk.NW,
                text="⚠️ FRAME-BILD FEHLT\n(Pfade prüfen)",
                fill="#ff6b6b",
                font=("Consolas", 11, "bold"),
            )

        # Rechts: Maske zeichnen
        self.canvas_mask.delete("all")
        if self.tk_mask:
            self.canvas_mask.create_image(
                0, 0, anchor=tk.NW, image=self.tk_mask
            )

        # Rechteck auf BEIDEN Ansichten zeichnen
        for canvas in [self.canvas_frame, self.canvas_mask]:
            if self.is_drawing:
                # Während dem Ziehen (Gelber gestrichelter Rahmen)
                canvas.create_rectangle(
                    self.start_x,
                    self.start_y,
                    self.current_x,
                    self.current_y,
                    outline="#ffff00",
                    width=2,
                    dash=(4, 2),
                )
            elif self.crop_box is not None:
                # Feste gesetzte Box (Grüner Rahmen mit Eckpunkten)
                x1, y1, x2, y2 = self.crop_box
                cx1, cy1 = x1 * self.scale, y1 * self.scale
                cx2, cy2 = x2 * self.scale, y2 * self.scale

                canvas.create_rectangle(
                    cx1, cy1, cx2, cy2, outline="#00ff00", width=2
                )

                # Eck-Markierungen
                for px, py in [
                    (cx1, cy1),
                    (cx2, cy1),
                    (cx2, cy2),
                    (cx1, cy2),
                ]:
                    canvas.create_rectangle(
                        px - 3, px - 3, px + 3, px + 3, fill="red"
                    )

                # Maßangabe im Bild anzeigen
                w_orig = x2 - x1
                h_orig = y2 - y1
                canvas.create_text(
                    cx1 + 5,
                    cy1 + 15,
                    anchor=tk.NW,
                    text=f"{w_orig}x{h_orig}",
                    fill="#00ff00",
                    font=("Arial", 10, "bold"),
                )

    def reset_box(self):
        self.crop_box = None
        self.is_drawing = False
        self.status_label.config(text="Rechteck mit Maus ziehen")
        self.btn_reset.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.redraw()

    def save_box(self):
        if self.crop_box is None:
            return

        x1, y1, x2, y2 = self.crop_box
        w = x2 - x1
        h = y2 - y1

        frame_tag = (
            f"{self.current_frame_nr:04d}"
            if self.current_frame_nr is not None
            else self.current_task
        )
        out_filename = f"{self.video_clean_name}_{frame_tag}_cut.txt"
        gt_folder_path = os.path.abspath(
            os.path.join(GT_DIR, self.current_folder_name)
        )
        out_path = os.path.join(gt_folder_path, out_filename)

        # Speichert die Crop-Box mit Slicing-Hilfe für TIDE
        with open(out_path, "w") as f:
            f.write(
                f"# Video: {self.video_clean_name}, Frame: {frame_tag}, Task: {self.current_task}\n"
            )
            f.write("# Format: x1, y1, x2, y2\n")
            f.write(f"{x1}, {y1}, {x2}, {y2}\n")
            f.write(f"# Box BxH: {w} x {h}\n")
            f.write(f"# Python Crop Slice: frame[{y1}:{y2}, {x1}:{x2}]\n")

        print(f"[SAVE] Gespeichert: {out_path} ➔ ({x1}, {y1}, {x2}, {y2})")
        messagebox.showinfo(
            "Gespeichert",
            f"Crop-Box erfolgreich gespeichert!\n\nDatei: {out_filename}\nKoordinaten: [{x1}, {y1}, {x2}, {y2}]\nPython Slice: frame[{y1}:{y2}, {x1}:{x2}]",
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = BoundingBoxCropAnnotatorGUI(root)
    root.mainloop()