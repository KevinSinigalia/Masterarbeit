import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= KONFIGURATION =================
INPUT_FOLDER = "./recordings_counted/frames_fishvideo67"
OUTPUT_FOLDER = "./recordings_counted/cropped_frames/frames_fishvideo67_cropped"
# =================================================

selected_points = []


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def create_tracker():
    """Versucht verschiedene OpenCV-Tracker-APIs zu initialisieren"""
    if hasattr(cv2, 'TrackerCSRT') and hasattr(cv2.TrackerCSRT, 'create'):
        return cv2.TrackerCSRT.create(), "CSRT"
    if hasattr(cv2, 'TrackerKCF') and hasattr(cv2.TrackerKCF, 'create'):
        return cv2.TrackerKCF.create(), "KCF"
    if hasattr(cv2, 'TrackerMIL') and hasattr(cv2.TrackerMIL, 'create'):
        return cv2.TrackerMIL.create(), "MIL"
    if hasattr(cv2, 'legacy'):
        if hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create(), "CSRT_legacy"
        if hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create(), "KCF_legacy"
    return None, "TEMPLATE_MATCHING"


def onclick(event):
    """Event-Handler für Mausklicks zur Auswahl der 4 Ecken"""
    global selected_points
    if event.xdata is not None and event.ydata is not None:
        # Linksklick: Punkt hinzufügen
        if event.button == 1 and len(selected_points) < 4:
            selected_points.append([int(event.xdata), int(event.ydata)])
            plt.plot(event.xdata, event.ydata, 'ro', markersize=6)
            plt.text(event.xdata + 5, event.ydata + 5, str(len(selected_points)), color='yellow', fontsize=12,
                     weight='bold')

            # Linien zwischen den Punkten zeichnen
            if len(selected_points) > 1:
                p1 = selected_points[-2]
                p2 = selected_points[-1]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
            if len(selected_points) == 4:
                # Letzte Linie zum Schließen des Trapezes
                plt.plot([selected_points[3][0], selected_points[0][0]], [selected_points[3][1], selected_points[0][1]],
                         'r-', linewidth=2)
                plt.title("4 Punkte gesetzt! Du kannst das Fenster jetzt SCHLIESSEN (X).")
            plt.draw()

        # Rechtsklick: Letzten Punkt entfernen
        elif event.button == 3 and len(selected_points) > 0:
            selected_points.pop()
            # Plot neu zeichnen
            ax.clear()
            ax.imshow(rgb_frame)
            for i, p in enumerate(selected_points):
                plt.plot(p[0], p[1], 'ro', markersize=6)
                plt.text(p[0] + 5, p[1] + 5, str(i + 1), color='yellow', fontsize=12, weight='bold')
                if i > 0:
                    plt.plot([selected_points[i - 1][0], p[0]], [selected_points[i - 1][1], p[1]], 'r-', linewidth=2)
            plt.title("Punkt entfernt. Klicke 4 Ecken an.")
            plt.draw()


def main():
    global rgb_frame, ax

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

    # 3. GUI zur Trapez-Auswahl (4 Punkte)
    print("\n--- BEDIENUNG ---")
    print(
        "1. Klicke mit LINKS nacheinander die 4 ECKEN des Aquariums an (z.B. Oben-Links -> Oben-Rechts -> Unten-Rechts -> Unten-Links).")
    print("2. Mit RECHTSKLICK kannst du den letzten Punkt löschen, falls du dich verklickt hast.")
    print("3. Wenn 4 Punkte gesetzt sind, SCHLIESSE das Fenster (X oben rechts).")
    print("-----------------\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb_frame)
    plt.title("Klicke nacheinander die 4 Ecken an (1->2->3->4)")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(selected_points) != 4:
        print("Es wurden nicht genau 4 Punkte ausgewählt. Abbruch.")
        return

    poly_pts = np.array(selected_points, dtype=np.int32)

    # Umschließende Bounding Box für das Tracking berechnen
    bx, by, bw, bh = cv2.boundingRect(poly_pts)
    current_box = [bx, by, bw, bh]
    initial_box_pos = np.array([bx, by])

    # 4. Tracking initialisieren
    tracker, tracker_type = create_tracker()
    print(f"Verwende Tracking-Methode: {tracker_type}")

    template = None
    if tracker is not None:
        tracker.init(first_frame, tuple(current_box))
    else:
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        template = first_gray[by:by + bh, bx:bx + bw]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 5. Alle Frames verarbeiten
    print("\nVerarbeite Frames mit Trapez-Maskierung...")
    for idx, path in enumerate(tqdm(frame_paths, desc="Fortschritt")):
        img = cv2.imread(path)
        if img is None:
            continue

        if idx > 0:
            if tracker is not None:
                success, bbox = tracker.update(img)
                if success:
                    current_box = [int(v) for v in bbox]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pad = 80
                sy1 = max(0, current_box[1] - pad)
                sy2 = min(img_h, current_box[1] + current_box[3] + pad)
                sx1 = max(0, current_box[0] - pad)
                sx2 = min(img_w, current_box[0] + current_box[2] + pad)

                search_area = gray[sy1:sy2, sx1:sx2]
                res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)

                current_box[0] = sx1 + max_loc[0]
                current_box[1] = sy1 + max_loc[1]

        # Verschiebung (Delta X, Delta Y) der Kamera berechnen
        dx = current_box[0] - initial_box_pos[0]
        dy = current_box[1] - initial_box_pos[1]

        # Die 4 Punkte des Trapezes um die Verschiebung mitbewegen
        shifted_poly = poly_pts + np.array([dx, dy])

        # 1-Kanal-Maske für das Trapez erzeugen
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_poly], 255)

        # Bild außerhalb des Trapezes schwarz färben
        masked_frame = cv2.bitwise_and(img, img, mask=mask)

        # Speichern
        filename = os.path.basename(path)
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, masked_frame)

    print(f"\nFertig! Trapez-gecroppte Frames gespeichert in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()