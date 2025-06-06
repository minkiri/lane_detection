import torch
from torch.utils.data import DataLoader
from lane_dataset import LaneDataset
from lanenet_model import LaneNet
import torch.nn as nn
from tqdm import tqdm

# --- 설정 ---
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/lanenet.pth"

# --- 데이터 로딩 ---
val_dataset = LaneDataset("dataset/val", "dataset/val_label")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 모델 로드 ---
model = LaneNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 손실 함수 ---
criterion = nn.BCEWithLogitsLoss()

# --- 검증 루프 ---
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="[Val]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        val_loss += loss.item()

        # 예측 마스크와 실제 마스크 비교
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == masks).sum().item()
        total += masks.numel()

    val_accuracy = correct / total
    print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {val_accuracy:.4f}")

