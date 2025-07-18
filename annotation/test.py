import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from RadarBBoxWeaponNet import RadarBBoxWeaponNet
from RadarPreprocessing import RadarPreprocessing

# === CONFIG ===
frame_id = 88
session_dir = r"C:/Users/Hp/Downloads/Dataset/2021_05_11_kf_oc003"
radar_path = os.path.join(session_dir, "radar_raw_frame", f"{frame_id:06d}.mat")
image_path = os.path.join(session_dir, "images_0", f"{frame_id:010d}.jpg")
label_file_path = os.path.join(session_dir, "label_2.0.txt")
model_path = r"C:/Users/Hp/Downloads/BBoxOutput/best_bbox_model_epoch_26.pth"
output_path = os.path.join(session_dir, f"frame_{frame_id:06d}_pred_vs_gt.jpg")
img_w, img_h = 1440, 1080

# === UTILITIES ===
def denormalize_bbox(norm_bbox, img_w, img_h):
    cx, cy, w, h = norm_bbox
    return [cx * img_w, cy * img_h, w * img_w, h * img_h]

def xywh_to_xyxy(cx, cy, w, h):
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
def xywh_to_xyxy2(cx, cy, w, h):
    return [cx , cy , w, h ]

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

# === LOAD GROUND TRUTH ===
gt_bbox = None
with open(label_file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if int(parts[0]) == frame_id:
            x1, y1, x2, y2 = map(float, parts[2:6])
            gt_bbox = [x1, y1, x2, y2]
            break

if gt_bbox is None:
    raise ValueError(f" No ground truth found for frame {frame_id}")

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RadarBBoxWeaponNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === PREPROCESS ===
preprocessor = RadarPreprocessing((32, 32, 16))
adc = preprocessor.preprocess(radar_path)
if adc is None:
    raise ValueError(f" Failed to preprocess radar file {radar_path}")

radar_tensor = torch.tensor(adc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# === INFERENCE ===
with torch.no_grad():
    bbox_pred, weapon_pred = model(radar_tensor)
    bbox_pred = bbox_pred.squeeze().cpu().numpy()
    weapon_score = weapon_pred.item()

# === PREDICTION BOX ===
cx, cy, w, h = denormalize_bbox(bbox_pred, img_w, img_h)
# Store cx, cy, w, h for display
pred_cx, pred_cy, pred_w, pred_h = cx, cy, w, h
# Convert only for drawing and IoU
pred_bbox = xywh_to_xyxy(pred_cx, pred_cy, pred_w, pred_h)
gt_bbox_xyxy =  xywh_to_xyxy(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3])
print(f" Pred (cx, cy, w, h): {[round(pred_cx, 1), round(pred_cy, 1), round(pred_w, 1), round(pred_h, 1)]}")


# === GT BOX STATS ===
#gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
#gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
#gt_w = abs(gt_bbox[2] - gt_bbox[0])
#gt_h = abs(gt_bbox[3] - gt_bbox[1])
#gt_box_cxcywh = [gt_cx, gt_cy, gt_w, gt_h]
#gt_bbox_xyxy = xywh_to_xyxy(gt_cx, gt_cy, gt_w, gt_h)


# === CALCULATE IoU ===
iou = compute_iou(pred_bbox, gt_bbox_xyxy)

# === DRAW ===
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Fix: sort GT box for drawing only (preserve original for IoU)
# gt_draw_box = [
#     min(gt_bbox[0], gt_bbox[2]),
#     min(gt_bbox[1], gt_bbox[3]),
#     max(gt_bbox[0], gt_bbox[2]),
#     max(gt_bbox[1], gt_bbox[3])
# ]
gt_draw_box = [
    (gt_bbox_xyxy[0]),
    (gt_bbox_xyxy[1]),
    (gt_bbox_xyxy[2]),
    (gt_bbox_xyxy[3])
]
pred_draw_box = [
    min(pred_bbox[0], pred_bbox[2]),
    min(pred_bbox[1], pred_bbox[3]),
    max(pred_bbox[0], pred_bbox[2]),
    max(pred_bbox[1], pred_bbox[3])
]

draw.rectangle(gt_draw_box, outline="green", width=2)
draw.rectangle(pred_draw_box, outline="red", width=2)
label = f"{'Weapon' if weapon_score > 0.5 else 'No Weapon'} ({weapon_score:.2f}) IoU={iou:.2f}"
draw.text((pred_draw_box[0], pred_draw_box[1] - 20), label, fill="red")
image.save(output_path)

# === PRINT RESULTS ===
print(f"\n🖼️ Frame {frame_id}")
print(f" GT BBox (x1, y1, x2, y2): {[int(x) for x in gt_bbox]}")
#print(f" GT (cx, cy, w, h): {[round(gt_cx, 1), round(gt_cy, 1), round(gt_w, 1), round(gt_h, 1)]}")
#print(f" Pred BBox (x1, y1, x2, y2): {[round(x, 1) for x in pred_bbox]}")
print(f" Pred (cx, cy, w, h): {[round(cx, 1), round(cy, 1), round(w, 1), round(h, 1)]}")
print(f"🔍 Weapon Confidence: {weapon_score:.4f}")
print(f"📐 IoU: {iou:.4f}")
print(f" Overlay saved at: {output_path}")
