import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from lanenet_model import LaneNet

# --- 설정 ---
IMAGE_PATH = "dataset/val/Town04_Clear_Noon_09_09_2020_14_57_22_frame_1538_validation_set.png"  # 테스트할 이미지 경로
MODEL_PATH = "checkpoints/lanenet.pth"  # 학습된 모델
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 전처리 ---
def preprocess(image_path):
    image = cv2.imread(image_path)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, (512, 256))  # 모델에 맞는 크기로 리사이즈 (학습 시 크기 확인 필요)

    # Tensor로 변환 후 정규화 처리 (선택적으로 정규화 추가 가능)
    tensor = transforms.ToTensor()(resized).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    return tensor, original

# --- 후처리 + 오버레이 ---
def postprocess(mask, original):
    mask = torch.sigmoid(mask).squeeze().detach().cpu().numpy()  # [H,W]
    binary = (mask > 0.5).astype(np.uint8) * 255  # 임계값 0.5로 변경 (이진화)

    binary = cv2.resize(binary, (original.shape[1], original.shape[0]))  # 원본 크기에 맞게 리사이즈
    overlay = original.copy()
    overlay[binary > 0] = [0, 255, 0]  # 초록색 차선

    return overlay

# --- 모델 로드 ---
model = LaneNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 추론 수행 ---
with torch.no_grad():
    input_tensor, original = preprocess(IMAGE_PATH)
    output = model(input_tensor)
    result = postprocess(output, original)

    # 결과 저장
    os.makedirs("output", exist_ok=True)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식으로 저장
    cv2.imwrite("output/result.png", result_bgr)
    print("완료: output/result.png")

