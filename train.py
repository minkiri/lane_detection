import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lane_dataset import LaneDataset
from lanenet_model import LaneNet
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import time

# --- 설정 ---
BATCH_SIZE = 4
EPOCHS = 20  # 더 많은 에폭을 시도할 수 있도록 설정
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENT = 3  # 성능 향상이 없는 최대 에폭 수 (즉, 5 에폭 동안 개선되지 않으면 중지)

# --- 데이터 로딩 ---
train_dataset = LaneDataset("dataset/train", "dataset/train_label")
val_dataset = LaneDataset("dataset/val", "dataset/val_label")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 모델, 손실, 옵티마이저 ---
model = LaneNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()  # 이진 분류용 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- 학습 루프 ---
best_val_loss = float("inf")  # 초기값을 무한대로 설정
epochs_without_improvement = 0  # 성능 향상이 없었던 에폭 수

# 손실 추적용 리스트
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # --- 검증 루프 ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    # 손실 추적
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    # 검증 손실 확인
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # 성능 개선이 없으면 학습 중지
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # 모델 저장
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/lanenet.pth")
        print(f"모델 저장 완료 → checkpoints/lanenet.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENT:
            print(f"조기 종료: {PATIENT} 에폭 동안 개선이 없었습니다.")
            break

# --- 훈련 및 검증 손실 그래프 시각화 ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# --- 예측 시각화 ---
def visualize_predictions(model, val_loader, device, num_images=4):  # num_images를 배치 크기와 맞춤
    model.eval()
    images, masks = next(iter(val_loader))  # 첫 번째 배치 가져오기
    images, masks = images.to(device), masks.to(device)
    outputs = model(images)

    # 결과 시각화
    for i in range(min(num_images, images.size(0))):  # 배치 크기만큼만 시각화
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # [H, W, C]
        true_mask = masks[i].cpu().numpy().squeeze()  # [H, W]
        predicted_mask = torch.sigmoid(outputs[i]).cpu().detach().numpy().squeeze()  # [H, W]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.show()


visualize_predictions(model, val_loader, DEVICE)

# --- 추론 속도 측정 (FPS) ---
def measure_inference_speed(model, val_loader, device):
    model.eval()
    start_time = time.time()
    num_batches = len(val_loader)

    for images, _ in val_loader:
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)

    end_time = time.time()
    total_time = end_time - start_time
    fps = num_batches * BATCH_SIZE / total_time
    print(f"Inference Speed: {fps:.2f} FPS")

measure_inference_speed(model, val_loader, DEVICE)

