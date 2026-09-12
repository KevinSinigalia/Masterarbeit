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
    rect=True,       # Hält das echte Hochkant-Format bei
    mosaic=0.0,      # <--- Deaktiviert das 4er-Mosaik komplett!
    mixup=0.0,       # Deaktiviert das Überblenden von Bildern
    degrees=5.0,     # Nur leichte Drehung
    batch=8
)

print("\n--- Training abgeschlossen! ---")