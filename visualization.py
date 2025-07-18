import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from RadarBBoxWeaponNet import RadarBBoxWeaponNet
from RadarPreprocessing import RadarPreprocessing

# === Helper Functions ===
def denormalize_bbox(norm_bbox, img_w, img_h):
    cx, cy, w, h = norm_bbox
    return [cx * img_w, cy * img_h, w * img_w, h * img_h]

def xywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_boxes(image_path, pred_box, gt_box, save_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(gt_box, outline="green", width=2)  # GT: green
    draw.rectangle(pred_box, outline="red", width=2)  # Pred: red
    img.save(save_path)

# === Config ===
session_dir = r"C:/Users/Hp/Downloads/Dataset_1/2021_05_11_kf_oc003"
radar_dir = os.path.join(session_dir, "radar_raw_frame")
image_dir = os.path.join(session_dir, "images_0")
label_file_path = os.path.join(session_dir, "label_2.0.txt")
model_path = r"C:/Users/Hp/Downloads/BBoxOutput/best_bbox_model_epoch_32.pth"
save_overlay_dir = os.path.join(session_dir, "overlay_results")
os.makedirs(save_overlay_dir, exist_ok=True)
save_csv = os.path.join(session_dir, "iou_report.csv")

img_w, img_h = 1440, 1080

# === Load GT Labels ===
gt_labels = {}
with open(label_file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        x1, y1, x2, y2 = map(float, parts[2:6])
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        gt_labels[frame_id] = [x1, y1, x2, y2]

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RadarBBoxWeaponNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Preprocessor ===
preprocessor = RadarPreprocessing((32, 32, 16))

# === Evaluation ===
ious = []
frame_ids = []
with open(save_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Frame ID", "IoU", "GT x1", "GT y1", "GT x2", "GT y2", "Pred x1", "Pred y1", "Pred x2", "Pred y2"])

    for frame_id, gt_xyxy in gt_labels.items():
        radar_path = os.path.join(radar_dir, f"{frame_id:06d}.mat")
        image_path = os.path.join(image_dir, f"{frame_id:010d}.jpg")
        if not (os.path.exists(radar_path) and os.path.exists(image_path)):
            continue

        adc = preprocessor.preprocess(radar_path)
        if adc is None:
            continue

        radar_tensor = torch.tensor(adc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            bbox_pred, _ = model(radar_tensor)
            bbox_pred = bbox_pred.squeeze().cpu().numpy()

        # Convert to pixel coords
        cx, cy, w, h = denormalize_bbox(bbox_pred, img_w, img_h)
        pred_xyxy = xywh_to_xyxy(cx, cy, w, h)

        iou = compute_iou(pred_xyxy, gt_xyxy)
        ious.append(iou)
        frame_ids.append(frame_id)

        writer.writerow([frame_id, iou] + gt_xyxy + pred_xyxy)

        # Save overlay image
        overlay_path = os.path.join(save_overlay_dir, f"frame_{frame_id:06d}_overlay.jpg")
        draw_boxes(image_path, pred_xyxy, gt_xyxy, overlay_path)

        print(f" Frame {frame_id} | IoU: {iou:.4f}")

# === Final Stats ===
ious = np.array(ious)
mean_iou = np.mean(ious)
above_05 = np.sum(ious > 0.5) / len(ious) * 100

print(f"\n Evaluation Report:")
print(f"    Total Samples Evaluated: {len(ious)}")
print(f"    Mean IoU: {mean_iou:.4f}")
print(f"    % Predictions with IoU > 0.5: {above_05:.2f}%")

# === Histogram ===
plt.hist(ious, bins=20, color='blue', alpha=0.7)
plt.title("IoU Distribution Across Test Samples")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
