from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
import cv2
import os

class PFRatioDataset(Dataset):
    def __init__(self, cfg, reverse=False, flag="train", transform=None):
        if flag == "train":
            df_path = os.path.join(cfg.data_root_path, cfg.dataset.train_csv_file)
        elif flag == "valid":
            df_path = os.path.join(cfg.data_root_path, cfg.dataset.valid_csv_file)
        elif flag == "test":
            df_path = os.path.join(cfg.data_root_path, cfg.dataset.test_csv_file)

        img_dirs = os.path.join(cfg.data_root_path, cfg.image_dirs)
        self.df = pd.read_csv(df_path)
        
        # 원본에서는 CXR PATH가 바론 ./data_png인데, 현재 디렉토리 구조간 ./data/data_png, ./data/metadata로 구성돼있어서 필요하면 사용
        self.df['CXR PATH1'] = self.df['CXR PATH1'].apply(lambda x: x.replace("./data_png", img_dirs))
        self.df['CXR PATH2'] = self.df['CXR PATH2'].apply(lambda x: x.replace("./data_png", img_dirs))
        
        if reverse:
            swap = self.df['CXR PATH1']
            self.df['CXR PATH1'] = self.df['CXR PATH2']
            self.df['CXR PATH2'] = swap
            
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x1 = cv2.imread(row["CXR PATH1"], cv2.IMREAD_GRAYSCALE)
        x2 = cv2.imread(row["CXR PATH2"], cv2.IMREAD_GRAYSCALE)

        y = torch.tensor(row["SIMPLE LABEL"], dtype=torch.float32) 
        
        pid = row["PID"]
        cxr_path1 = row['CXR PATH1']
        cxr_path2 = row['CXR PATH2']

        if self.transform is not None:
            x1 = self.transform(x1)  
            x2 = self.transform(x2)
            
        res = {
            "x1": x1,
            "x2": x2,
            "y": y,
            "pid": pid,
            "cxr_path1": cxr_path1,
            "cxr_path2": cxr_path2
        }

        return res