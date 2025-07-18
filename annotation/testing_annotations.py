import os
import gc
import torch
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from RadarBBoxWeaponNet import RadarBBoxWeaponNet
from RadarPreprocessing import RadarPreprocessing

# === CONFIG ===
session_dir = r"C:/Users/Hp/Downloads/Dataset/2021_05_11_kf_oc003"
label_file_path = os.path.join(session_dir, "label_2.0.txt")
model_path = r"C:/Users/Hp/Downloads/BBoxOutput/best_bbox_model_epoch_26.pth"
output_dir = os.path.join(session_dir, "bbox_all_outputs")
os.makedirs(output_dir, exist_ok=True)
img_w, img_h = 1440, 1080

# === UTILITIES ===
def denormalize_bbox(norm_bbox, img_w, img_h):
    cx, cy, w, h = norm_bbox
    return [cx * img_w, cy * img_h, w * img_w, h * img_h]

def xywh_to_xyxy(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

def compute_iou(boxA, boxB):
    xA = max(min(boxA[0], boxA[2]), min(boxB[0], boxB[2]))
    yA = max(min(boxA[1], boxA[3]), min(boxB[1], boxB[3]))
    xB = min(max(boxA[0], boxA[2]), max(boxB[0], boxB[2]))
    yB = min(max(boxA[1], boxA[3]), max(boxB[1], boxB[3]))
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RadarBBoxWeaponNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === PREPROCESSOR ===
preprocessor = RadarPreprocessing((32, 32, 16))

# === LOAD GROUND TRUTH BOXES ===
gt_dict = {}
with open(label_file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        frame_id = int(parts[0])
        x1, y1, x2, y2 = map(float, parts[2:6])
        gt_dict[frame_id] = [x1, y1, x2, y2]

# === RESULTS LIST ===
results = []

# === LOOP THROUGH FRAMES ===
for frame_id, gt_bbox in gt_dict.items():
    radar_path = os.path.join(session_dir, "radar_raw_frame", f"{frame_id:06d}.mat")
    image_path = os.path.join(session_dir, "images_0", f"{frame_id:010d}.jpg")
    output_path = os.path.join(output_dir, f"frame_{frame_id:06d}_pred_vs_gt.jpg")

    if not os.path.exists(radar_path) or not os.path.exists(image_path):
        continue

    adc = preprocessor.preprocess(radar_path)
    if adc is None:
        continue

    radar_tensor = torch.tensor(adc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        bbox_pred, weapon_pred = model(radar_tensor)
        bbox_pred = bbox_pred.squeeze().cpu().numpy()
        weapon_score = weapon_pred.item()

    # Denormalize and convert to xyxy format
    cx, cy, w, h = denormalize_bbox(bbox_pred, img_w, img_h)
    pred_bbox = xywh_to_xyxy(cx, cy, w, h)
    gt_bbox_xyxy = xywh_to_xyxy(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3])
    iou = compute_iou(pred_bbox, gt_bbox_xyxy)

    # Draw on image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    pred_draw_box = [
        min(pred_bbox[0], pred_bbox[2]), min(pred_bbox[1], pred_bbox[3]),
        max(pred_bbox[0], pred_bbox[2]), max(pred_bbox[1], pred_bbox[3])
    ]
    gt_draw_box = [
        min(gt_bbox_xyxy[0], gt_bbox_xyxy[2]), min(gt_bbox_xyxy[1], gt_bbox_xyxy[3]),
        max(gt_bbox_xyxy[0], gt_bbox_xyxy[2]), max(gt_bbox_xyxy[1], gt_bbox_xyxy[3])
    ]
    draw.rectangle(gt_draw_box, outline="green", width=2)
    draw.rectangle(pred_draw_box, outline="red", width=2)
    label = f"{'Weapon' if weapon_score > 0.5 else 'No Weapon'} ({weapon_score:.2f}) IoU={iou:.2f}"
    draw.text((pred_draw_box[0], pred_draw_box[1] - 20), label, fill="red")
    image.save(output_path)

    results.append({
        "frame_id": frame_id,
        "gt_cx": gt_bbox[0], "gt_cy": gt_bbox[1], "gt_w": gt_bbox[2], "gt_h": gt_bbox[3],
        "pred_cx": cx, "pred_cy": cy, "pred_w": w, "pred_h": h,
        "iou": iou,
        "weapon_score": weapon_score
    })

    # Cleanup
    del radar_tensor, adc, image, draw
    torch.cuda.empty_cache()
    gc.collect()

# === SAVE RESULTS TO CSV ===
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "bbox_iou_results.csv")
df.to_csv(csv_path, index=False)
print(f" All results saved at: {csv_path}")
from sklearn.metrics import precision_recall_curve, auc

# --- Prepare data ---
all_scores = []
all_labels = []

for res in results:
    iou = res["iou"]
    score = res["weapon_score"]
    is_tp = iou >= 0.5  # True Positive if IoU >= 0.5

    all_scores.append(score)
    all_labels.append(1 if is_tp else 0)

# --- Compute Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(all_labels, all_scores)
ap = auc(recall, precision)

print(f"\n mAP@0.5 (single class weapon): {ap:.4f}")
