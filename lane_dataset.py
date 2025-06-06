import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LaneDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.label_paths = [os.path.join(label_dir, fname.replace('.png', '_label.png')) for fname in os.listdir(image_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_labels(self, label_image):
        # 라벨 값 1, 2를 1로 변경 (이진화)
        label_image[label_image == 1] = 1
        label_image[label_image == 2] = 1
        label_image[label_image == 0] = 0
        return label_image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 이미지 읽기
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 라벨은 그레이스케일로 읽음

        # 이미지 및 라벨 리사이징
        image = cv2.resize(image, (512, 256))
        label = cv2.resize(label, (512, 256))

        # 라벨 값 1, 2를 1로 변경
        label = self.preprocess_labels(label)

        # 이미지와 라벨을 Tensor로 변환
        image = transforms.ToTensor()(image)  # [3,H,W], 0~1
        label = torch.from_numpy(label).unsqueeze(0).float()  # [1,H,W], 0~1 (이진화)

        # 필요시 이미지 정규화
        if self.transform:
            image = self.transform(image)

        return image, label

