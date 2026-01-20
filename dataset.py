from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
import cv2

class PFRatioDataset(Dataset):
    def __init__(self, cfg, flag="train", transform=None):
        self.patient_info = pd.read_csv(cfg.csv_path)
        rename_df = pd.read_excel(cfg.renamed_excel_path)  # 필요시 엑셀 파일도 읽기
        
        # 원본에서는 CXR PATH가 바론 ./data_png인데, 현재 디렉토리 구조간 ./data/data_png, ./data/metadata로 구성돼있어서 필요하면 사용
        self.patient_info['CXR PATH1'] = self.patient_info['CXR PATH1'].apply(lambda x: x.replace("./data_png", cfg.data_root_path))
        self.patient_info['CXR PATH2'] = self.patient_info['CXR PATH2'].apply(lambda x: x.replace("./data_png", cfg.data_root_path))
        
        self.patient_info['SIMPLE LABEL'] = self.patient_info['SIMPLE LABEL'].apply(lambda x: 1 if x == 2 else 0)
        
        
        self.transform = transform

    def __len__(self):
        return len(self.patient_info)
    
    def __getitem__(self, idx):
        row = self.patient_info.iloc[idx]

        x1 = cv2.imread(row["CXR PATH1"], cv2.IMREAD_GRAYSCALE)
        x2 = cv2.imread(row["CXR PATH2"], cv2.IMREAD_GRAYSCALE)

        y = torch.tensor(row["SIMPLE LABEL"], dtype=torch.float32)
        
        pid = row["PID"]

        if self.transform is not None:
            x1 = self.transform(x1)  
            x2 = self.transform(x2)
            
        res = {
            "x1": x1,
            "x2": x2,
            "y": y,
            "pid": pid
        }

        return res