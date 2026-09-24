import subprocess
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "outputs_raw"  # Pfad zu deinen Ergebnissen


def run_processing():
    # Einstellungen
    FIRST_VIDEO = 30
    LAST_VIDEO = 30
    DURATION = 40  # Länge eines Segments
    OVERLAP = 5  # Überlappung (n = n + duration - 5)

    python_executable = "python3"  # oder "python3" oder dein venv Pfad

    for vid in range(FIRST_VIDEO, LAST_VIDEO + 1):
        frames_dir = Path(f"videos/frames_fishvideo{vid}")

        # 1. Zählen, wie viele Frames im Ordner sind
        if not frames_dir.exists():
            print(f"Ordner {frames_dir} nicht gefunden, überspringe...")
            continue

        # Zählt alle .jpg oder .png Dateien im Ordner
        total_frames = len(list(frames_dir.glob("*.jpg")))
        if total_frames == 0:
            total_frames = len(list(frames_dir.glob("*.png")))

        print(f"\nVerarbeite Video {vid} ({total_frames} Frames gefunden)")

        n = 1
        while True:
            start_f = n
            end_f = n + DURATION - 1

            # Falls das Fenster über das Ende des Videos hinausgehen würde,
            # wird die Schleife für dieses Video beendet.
            if end_f > total_frames:
                print(f"  -> Segment {start_f} bis {end_f} überschreitet total_frames ({total_frames}). Video beendet.")
                break

                # Den Befehl für den neuen Prozess zusammenbauen
            cmd = [
                # python_executable, "sam3_script_nils_edited_multi.py",
                python_executable, "sam3_masks_output.py",
                "--video_nr", str(vid),
                "--start", str(start_f),
                "--end", str(end_f)
            ]

            print(f"  -> Starte Prozess: Frames {start_f} bis {end_f}")

            # Prozess ausführen und auf Ende warten
            subprocess.run(cmd)

            # Logik für den nächsten Startpunkt: n = n + duration - 5
            # Da n am Anfang start_f war, ist das neue n:
            n = end_f - OVERLAP


if __name__ == "__main__":
    run_processing()