import os
import cv2
import shutil
from insightface.app import FaceAnalysis

TARGET_SIZE = (72, 72)
PADDING = 20

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def move_to_inconsistent(frame_path, session_path):
    inconsistent_dir = os.path.join(session_path, 'inconsistent_frames_folder')
    os.makedirs(inconsistent_dir, exist_ok=True)
    shutil.move(frame_path, os.path.join(inconsistent_dir, os.path.basename(frame_path)))

def process_frame(image_path, session_path, size=TARGET_SIZE, padding=PADDING):
    img = cv2.imread(image_path)
    if img is None:
        print(f" Failed to load: {image_path}")
        move_to_inconsistent(image_path, session_path)
        return False

    faces = app.get(img)
    if not faces:
        print(f" No face detected: {image_path}")
        move_to_inconsistent(image_path, session_path)
        return False

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)

    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Crop, resize, save
    face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, face_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return True

def perform_face_detection(base_dir):
    for folder in sorted(os.listdir(base_dir)):
        session_path = base_dir+'/'+ folder
        if not os.path.isdir(session_path):
            continue

        for subfolder in os.listdir(session_path):
            if subfolder.endswith("_frames"):
                frame_dir = os.path.join(session_path, subfolder)
                print(f"\n Processing: {frame_dir}")
                for frame in sorted(os.listdir(frame_dir)):
                    if frame.endswith(".png"):
                        frame_path = os.path.join(frame_dir, frame)
                        success = process_frame(frame_path, session_path)
                        if success:
                            print(f" Processed: {frame_path}")
                        else:
                            print(f" Skipped (moved): {frame_path}")

# if __name__ == "__main__":
#     perform_face_detection(BASE_DIR)
