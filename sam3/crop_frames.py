import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm

# ================= KONFIGURATION =================
INPUT_FOLDER = "./frames_fishvideo39"
OUTPUT_FOLDER = "./frames_fishvideo39_tracked"
# =================================================

roi_coords = None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def onselect(eclick, erelease):
    global roi_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    xmin = max(0, min(x1, x2))
    xmax = max(x1, x2)
    ymin = max(0, min(y1, y2))
    ymax = max(y1, y2)

    w = xmax - xmin
    h = ymax - ymin

    roi_coords = (xmin, ymin, w, h)


def create_tracker():
    """Versucht verschiedene OpenCV-Tracker-APIs zu initialisieren"""
    # 1. Neue OpenCV API (cv2.TrackerCSRT.create)
    if hasattr(cv2, 'TrackerCSRT') and hasattr(cv2.TrackerCSRT, 'create'):
        return cv2.TrackerCSRT.create(), "CSRT"
    if hasattr(cv2, 'TrackerKCF') and hasattr(cv2.TrackerKCF, 'create'):
        return cv2.TrackerKCF.create(), "KCF"
    if hasattr(cv2, 'TrackerMIL') and hasattr(cv2.TrackerMIL, 'create'):
        return cv2.TrackerMIL.create(), "MIL"

    # 2. Legacy OpenCV API
    if hasattr(cv2, 'legacy'):
        if hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create(), "CSRT_legacy"
        if hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create(), "KCF_legacy"

    # 3. Falls kein Tracker-Modul vorhanden -> Template Matching Modus
    return None, "TEMPLATE_MATCHING"


def main():
    # 1. Frames suchen
    extensions = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG')
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    frame_paths = [
        f for f in frame_paths
        if not os.path.splitext(os.path.basename(f))[0].endswith('_cropped')
    ]
    frame_paths = sorted(frame_paths, key=natural_sort_key)

    if not frame_paths:
        print(f"Fehler: Keine JPG-Bilder in '{INPUT_FOLDER}' gefunden!")
        return

    print(f"{len(frame_paths)} Frames gefunden.")

    # 2. Ersten Frame laden
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print("Fehler beim Laden des ersten Frames.")
        return

    img_h, img_w = first_frame.shape[:2]
    rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # 3. GUI zur Auswahl des Aquariums
    print("\n--- BEDIENUNG ---")
    print("1. Kasten GENAU um das mittlere Aquarium ziehen.")
    print("2. Danach das Fenster oben rechts über das X SCHLIESSEN.")
    print("-----------------\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb_frame)
    plt.title("Aquarium auswählen -> Danach Fenster SCHLIESSEN")

    rect_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    plt.show()

    if roi_coords is None or roi_coords[2] == 0 or roi_coords[3] == 0:
        print("Kein Bereich ausgewählt. Abbruch.")
        return

    x, y, w, h = roi_coords
    current_box = [x, y, w, h]

    # 4. Tracking-Methode vorbereiten
    tracker, tracker_type = create_tracker()
    print(f"Verwende Tracking-Methode: {tracker_type}")

    # Falls wir Template Matching nutzen, speichern wir den ersten Ausschnitt
    template = None
    if tracker is not None:
        tracker.init(first_frame, tuple(current_box))
    else:
        # Graustufen-Template des Aquariums für schnellen Bildabgleich
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        template = first_gray[y:y + h, x:x + w]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 5. Tracking über alle Frames
    print("\nVerarbeite Frames...")
    for idx, path in enumerate(tqdm(frame_paths, desc="Fortschritt")):
        img = cv2.imread(path)
        if img is None:
            continue

        if idx > 0:
            if tracker is not None:
                # OpenCV Tracker
                success, bbox = tracker.update(img)
                if success:
                    current_box = [int(v) for v in bbox]
            else:
                # Template-Matching Fallback
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Suchfenster auf die Umgebung des letzten Frames begrenzen (schneller & robuster)
                pad = 80  # Erlaubtes Wackeln in Pixeln
                sy1 = max(0, current_box[1] - pad)
                sy2 = min(img_h, current_box[1] + current_box[3] + pad)
                sx1 = max(0, current_box[0] - pad)
                sx2 = min(img_w, current_box[0] + current_box[2] + pad)

                search_area = gray[sy1:sy2, sx1:sx2]
                res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)

                current_box[0] = sx1 + max_loc[0]
                current_box[1] = sy1 + max_loc[1]

        # Koordinaten absichern
        cx, cy, cw, ch = current_box
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        cw = max(1, min(cw, img_w - cx))
        ch = max(1, min(ch, img_h - cy))

        # Außerhalb schwarz färben
        black_frame = np.zeros_like(img)
        black_frame[cy:cy + ch, cx:cx + cw] = img[cy:cy + ch, cx:cx + cw]

        filename = os.path.basename(path)
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, black_frame)

    print(f"\nFertig! Frames gespeichert in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()