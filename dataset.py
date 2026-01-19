from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
import cv2

class PFRatioDataset(Dataset):
    def __init__(self, cfg, flag="train", transform=None):
        self.patient_info = pd.read_csv(cfg.csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.patient_info)

    # def _read_gray_pil(self, path: str):
    #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #     if img is None:
    #         raise FileNotFoundError(f"Failed to read image: {path}")
    #     return Image.fromarray(img)  # PIL로 변환

    def __getitem__(self, idx):
        row = self.patient_info.iloc[idx]

        x1 = cv2.imread(row["CXR PATH1"], cv2.IMREAD_GRAYSCALE)
        x2 = cv2.imread(row["CXR PATH2"], cv2.IMREAD_GRAYSCALE)

        y = int(row["SIMPLE LABEL"])
        y = torch.tensor(row["SIMPLE LABEL"], dtype=torch.float32)

        if self.transform is not None:
            x1 = self.transform(x1)  
            x2 = self.transform(x2)

        return (x1, x2), y