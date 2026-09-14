import torch
from ultralytics import YOLO

# 2. Prüfen, ob GPU (CUDA) verfügbar ist
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Verwendetes Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")

# 3. Vortrainiertes Segmentierungs-Modell laden
# Optionen für Größe: yolov8n-seg (Nano), yolov8s-seg (Small), yolov8m-seg (Medium)
# Alternativ das neuere: yolo11n-seg.pt
model = YOLO("yolo26n-seg.pt")

# 4. Training starten
results = model.train(
    data="./dataset/data.yaml",
    epochs=300,
    imgsz=1024,
    batch=8,
    rect=True,  # Behält das schmale Hochkant-Format bei
    # --- DEAKTIVIERTE AUGMENTATIONEN ---
    mosaic=0.0,  # 4er-Mosaik komplett aus
    mixup=0.0,  # Überblenden von Bildern aus
    cutmix=0.0,  # CutMix explizit aus
    copy_paste=0.0,  # Copy-Paste von Objekten aus
    # --- AKTIVIERTE & ANGEPASSTE AUGMENTATIONEN ---
    degrees=180.0,  # Rotation von -180° bis +180° (alle Winkel)
    fliplr=0.5,  # Horizontales Spiegeln (50 % Wahrscheinlichkeit)
    flipud=0.0,  # Vertikales Spiegeln (auf dem Kopf stehend) auslassen
    erasing=0.4,  # Random Erasing / Cutout (40 % Wahrscheinlichkeit)
    # --- WEITERE TRAININGSEINSTELLUNGEN ---
    patience=30,  # Early Stopping: Stoppt, wenn 30 Epochen keine Verbesserung eintritt
    save=True,
    plots=True,
)

print("\n--- Training abgeschlossen! ---")