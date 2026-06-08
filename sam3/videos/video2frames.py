import cv2
import os

def extract_frames_opencv(video_path, output_folder, target_fps=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Berechne, jeden wievielten Frame wir nehmen müssen
    hop = source_fps / target_fps
    
    count = 0
    frame_id = 1 # Start bei 0001
    
    print(f"Video FPS: {source_fps}, Ziel FPS: {target_fps}")

    while True:
        # Springe zum richtigen Frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(count * hop))
        ret, frame = cap.read()
        
        if not ret or frame_id > 9999:
            break
            
        # Optional: Hier direkt verkleinern, um VRAM zu sparen!
        frame = cv2.resize(frame, (int(frame.shape[1] * (480/frame.shape[0])), 480))

        filename = os.path.join(output_folder, f"{frame_id:04d}.jpg")
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        if frame_id % 100 == 0:
            print(f"Frame {frame_id} extrahiert...")
            
        frame_id += 1
        count += 1

    cap.release()
    print("Fertig!")

# --- ANWENDUNG ---
extract_frames_opencv("./fishvideo2.mp4", "./frames")