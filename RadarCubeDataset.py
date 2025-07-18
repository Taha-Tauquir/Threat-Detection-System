import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RadarBBoxDataset import RadarBBoxDataset
from RadarBBoxWeaponNet import RadarBBoxWeaponNet
from tqdm import tqdm
import csv

def train_bbox_weapon_model(data_path, save_dir):
    dataset = RadarBBoxDataset(data_path)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)  # batch size lowered for clearer output
    val_loader = DataLoader(val_set, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarBBoxWeaponNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    loss_bbox = nn.SmoothL1Loss()
    loss_weapon = nn.BCELoss()

    bbox_loss_weight = 3.0
    weapon_loss_weight = 1.0

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Epoch", "TrainLoss", "ValLoss", "TrainBBoxLoss", "TrainWeaponLoss", "ValBBoxLoss", "ValWeaponLoss"])

    for epoch in range(1, 101):
        print(f"\n📘 === Epoch {epoch} ===")
        model.train()
        total_train_loss = 0
        total_train_bbox = 0
        total_train_weapon = 0

        for batch_idx, (x, bbox_target, weapon_label) in enumerate(train_loader):
            x = x.to(device)
            bbox_target = bbox_target.to(device)
            weapon_label = weapon_label.to(device)

            optimizer.zero_grad()
            bbox_pred, weapon_pred = model(x)

            loss1 = loss_bbox(bbox_pred, bbox_target)
            loss2 = loss_weapon(weapon_pred.view(-1), weapon_label.view(-1))
            loss = bbox_loss_weight * loss1 + weapon_loss_weight * loss2

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_bbox += loss1.item()
            total_train_weapon += loss2.item()

            # 🚨 Print batch debug info
            print(f"\n🔁 Batch {batch_idx+1}")
            for i in range(len(x)):
                print(f"   🧾 Sample {i+1}")
                print(f"      📌 GT Weapon Label: {weapon_label[i].item():.1f} | 🔮 Pred: {weapon_pred[i].item():.4f}")
                print(f"       GT BBox: {bbox_target[i].cpu().numpy()}")
                print(f"       Pred BBox: {bbox_pred[i].detach().cpu().numpy()}")

        # === Validation ===
        model.eval()
        total_val_loss = 0
        total_val_bbox = 0
        total_val_weapon = 0

        with torch.no_grad():
            for x, bbox_target, weapon_label in val_loader:
                x = x.to(device)
                bbox_target = bbox_target.to(device)
                weapon_label = weapon_label.to(device)

                bbox_pred, weapon_pred = model(x)

                val_loss1 = loss_bbox(bbox_pred, bbox_target)
                val_loss2 = loss_weapon(weapon_pred.view(-1), weapon_label.view(-1))
                val_loss = bbox_loss_weight * val_loss1 + weapon_loss_weight * val_loss2

                total_val_loss += val_loss.item()
                total_val_bbox += val_loss1.item()
                total_val_weapon += val_loss2.item()

        # === Logging ===
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_bbox = total_train_bbox / len(train_loader)
        avg_train_weapon = total_train_weapon / len(train_loader)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_bbox = total_val_bbox / len(val_loader)
        avg_val_weapon = total_val_weapon / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"\n Epoch {epoch}")
        print(f"   🔧 Train Loss: {avg_train_loss:.4f} | BBox: {avg_train_bbox:.4f} | Weapon: {avg_train_weapon:.4f}")
        print(f"    Val   Loss: {avg_val_loss:.4f} | BBox: {avg_val_bbox:.4f} | Weapon: {avg_val_weapon:.4f}")

        with open(log_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                epoch, avg_train_loss, avg_val_loss,
                avg_train_bbox, avg_train_weapon,
                avg_val_bbox, avg_val_weapon
            ])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_bbox_model_epoch_{epoch}.pth'))
            print(" Best model saved.")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break


# === Run ===
if __name__ == "__main__":
    train_bbox_weapon_model(
        r"C:/Users/Hp/Downloads/Dataset_1", 
        r"C:/Users/Hp/Downloads/BBoxOutput"
    )
