import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from lanenet_model import LaneNet
from lane_dataset import LaneDataset
import time  # 추가된 부분

def evaluate(model, val_loader, device):
    model.eval()
    
    # 각 클래스에 대해 IoU 계산을 위한 변수 초기화
    intersection = 0
    union = 0
    pixel_accuracy = 0
    total_pixels = 0
    correct_pixels = 0
    
    # 혼동 행렬 계산을 위한 리스트 초기화
    all_preds = []
    all_labels = []
    
    # FPS 측정을 위한 시간 초기화
    start_time = time.time()  # 추론 시작 시간
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            # 모델의 예측
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # sigmoid를 통해 확률값으로 변환
            preds = outputs > 0.5  # 0.5 이상의 확률을 가진 부분은 차선으로 예측

            # preds와 masks를 정수형으로 변환
            preds = preds.int()  # 예측 결과를 정수형으로 변환
            masks = masks.int()  # 마스크를 정수형으로 변환

            # 정확도 계산
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
            
            # IoU 계산을 위한 교차와 합집합 계산
            intersection += (preds & masks).sum().item()
            union += (preds | masks).sum().item()
            
            # 예측값과 실제 라벨을 리스트로 저장 (Confusion Matrix 계산을 위해)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())
    
    # 평균 IoU (mIoU) 계산
    iou = intersection / union if union != 0 else 0
    mIoU = iou  # 이진 분류라서 mIoU는 IoU와 동일

    # 전체 정확도 (Pixel Accuracy) 계산
    pixel_accuracy = correct_pixels / total_pixels

    # 결과 출력
    print(f"Pixel Accuracy: {pixel_accuracy * 100:.2f}%")
    print(f"IoU: {iou * 100:.2f}%")
    print(f"mIoU: {mIoU * 100:.2f}%")
    
    # Confusion Matrix 계산
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    cm = confusion_matrix(all_labels.flatten(), all_preds.flatten())

    # Confusion Matrix 출력 (True Positive, False Positive, False Negative, True Negative)
    print("Confusion Matrix:")
    print(cm)

    # FPS 측정을 위한 시간 계산
    end_time = time.time()  # 추론 끝 시간
    total_time = end_time - start_time  # 총 소요 시간
    num_batches = len(val_loader)
    fps = num_batches * BATCH_SIZE / total_time  # FPS 계산
    print(f"Inference Speed: {fps:.2f} FPS")

    # 결과표 형식으로 출력 (예시)
    results = {
        "Pixel Accuracy": f"{pixel_accuracy * 100:.2f}%",
        "IoU": f"{iou * 100:.2f}%",
        "mIoU": f"{mIoU * 100:.2f}%",
        "Inference Speed (FPS)": f"{fps:.2f}"
    }
    
    return results


# 평가 함수 호출 예시
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로딩
val_dataset = LaneDataset("dataset/val", "dataset/val_label")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 로드
model = LaneNet().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/lanenet.pth", map_location=DEVICE))

# 평가 수행
results = evaluate(model, val_loader, DEVICE)

# 논문 결과표 예시
print("Results for Evaluation:")
for key, value in results.items():
    print(f"{key}: {value}")

