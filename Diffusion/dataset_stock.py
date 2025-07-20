import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class StockDataset(Dataset):
    def __init__(self, csv_path, flag='train', size=None, data_mode='each_stock', symbol='GOOG', scale=True, reference = None, top_k = 1):
        """
        size: [seq_len, label_len, pred_len]
        data_mode: 'each_stock', 'all_stock', 'industry'
        symbol: tên của cổ phiếu (input luôn là GOOG)
        """
        if size is None:
            raise ValueError("cung cấp giá trị size, ví dụ: [30, 0, 7] (history 30 ngày, pred 7 ngày)")
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_mode = data_mode
        self.symbol = symbol
        self.scale = scale
        self.flag = flag
        self.top_k = top_k
        self.reference = reference


        df = pd.read_csv(csv_path, usecols=['Date', 'Close'])
        df.columns = df.columns.str.strip()
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        df_input = df.sort_values('Date')

        # Lưu lại cột Date cho input
        self.dates = pd.to_datetime(df_input['Date'], utc=True)

        # Lấy giá đóng cửa của GOOG làm input
        data = df_input['Close'].values.reshape(-1, 1)
        self.raw_data = data.astype(np.float32)
        self.dim = self.raw_data.shape[1]


        total_len = len(self.raw_data)
        train = int(total_len * 0.7)
        
        print(f"Train train {train}")

        border1s = [0, int(total_len * 0.7) - self.seq_len, int(total_len * 0.8) - self.seq_len]  # 0.8 = 0.7 + 0.1
        border2s = [int(total_len * 0.7), int(total_len * 0.8), total_len]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        self.data = self.raw_data[border1:border2]

        # Scaling: chuẩn hóa theo tập train (sử dụng toàn bộ raw_data cho scaling)
        self.scaler = StandardScaler()
        if self.scale:
            train_data = self.raw_data[0:border2s[0]]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data)
            self.data_x = data[border1:border2]
        
        self.reference = torch.clamp(self.reference, min=0, max=self.data.shape[0] - self.seq_len - self.pred_len)
    def __getitem__(self, index): 
        
        s_begin = index 
        s_end = s_begin + self.seq_len + self.pred_len
        # r_begin = s_end - self.label_len  # 20--40   40-->60
        r_begin = s_end - self.pred_len
        # r_end = r_begin + self.label_len + self.pred_len 
        r_end = r_begin + self.label_len + self.pred_len 
        reference = np.zeros((self.top_k * self.pred_len, self.dim))
        # reference = torch.zeros((self.top_k * self.pred_len, self.dim), dtype=torch.float32)


       # load ref 
        if self.flag == 'train':
            for i in range(self.top_k): #val :
                
                start_idx = (self.top_k * index + i) * self.pred_len #0 (0, 23)
                end_idx = (self.top_k * index + i + 1) * self.pred_len #24
                segment = self.reference[start_idx:end_idx]  # (24,)
                normalized = (segment - segment.min()) / (segment.max() - segment.min() + 1e-8)
                reference[i * self.pred_len : (i + 1) * self.pred_len] = normalized.unsqueeze(1)  # (24, 1)

        else:
            total_len = len(self.raw_data)
            train = int(total_len * 0.7)
            sample_train = train - self.seq_len - self.pred_len + 1

            his_index = sample_train + self.pred_len + self.seq_len - 1
            index = his_index + index  # a vt gọn index = sample_train cx đc

            for i in range(self.top_k): #val :
                
                start_idx = (self.top_k * index + i) * self.pred_len #0 (0, 23)
                end_idx = (self.top_k * index + i + 1) * self.pred_len #24
                segment = self.reference[start_idx:end_idx]  # (24,)
                normalized = (segment - segment.min()) / (segment.max() - segment.min() + 1e-8)
                reference[i * self.pred_len : (i + 1) * self.pred_len] = normalized.unsqueeze(1)  # (24, 1)
        # print(reference.shape)
        # start_idx = self.top_k * index  * self.pred_len 
        # end_idx = self.top_k * (index + 1) * self.pred_len
        # reference = self.reference[start_idx : end_idx]
        reference = reference.squeeze()
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        # Tạo mốc thời gian (timepoints) dựa trên index (có thể điều chỉnh tùy theo logic)
        timepoints = np.arange(self.seq_len + self.pred_len) * 1.0
        # Danh sách các feature id (số lượng feature = self.dim)
        feature_id = np.arange(self.dim) * 1.0
        
        # Tạo mask: observed_mask là mảng toàn 1
        observed_mask = np.ones_like(seq_x)
        # gt_mask được copy từ observed_mask nhưng phần dự báo (last pred_len) được đặt thành 0
        gt_mask = observed_mask.copy()
        gt_mask[-self.pred_len:] = 0.
        
        sample = {
            'observed_data': seq_x, 
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': timepoints,
            'feature_id': feature_id,
            'reference': reference,
            #'reference_image': reference_image
        }
        return sample
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataloader(csv_path, data_mode='all_stock', symbol='GOOG', size=None, batch_size=16, reference = None, top_k = 1):
    if size is None:
        size = [30, 0, 7]
    train_dataset = StockDataset(csv_path, flag='train', size=size, data_mode=data_mode, symbol=symbol, reference = reference, top_k = top_k)
    valid_dataset = StockDataset(csv_path, flag='val', size=size, data_mode=data_mode, symbol=symbol, reference = reference,  top_k = top_k)
    test_dataset = StockDataset(csv_path, flag='test', size=size, data_mode=data_mode, symbol=symbol, reference = reference,  top_k = top_k)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader



