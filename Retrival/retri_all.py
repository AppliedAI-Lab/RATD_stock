# xong AMZN
# chạy AMZN 9624
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import torch
print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

from PIL import Image
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64):
        super().__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation, padding, dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, emb_dim)

    def forward(self, x):
        feat = self.network(x)               # (B, C, T)
        feat = torch.mean(feat, dim=2)       # Global average pooling -> (B, C)
        emb = self.proj(feat)                # => (B, emb_dim)
        return emb
def create_sliding_windows(series, window_size, stride=1):
    return np.array([series[i:i+window_size] for i in range(0, len(series)-window_size+1, stride)])

def create_sliding_windows(series, window_size, stride=1):
    return np.array([series[i:i+window_size] for i in range(0, len(series)-window_size+1, stride)])

def main():
    csv_path = "sp500_industry.csv"
    seq_len = 96
    pred_len = 24
    window_size = seq_len + pred_len  # Tổng cộng 120 ngày
    top_k_list = [1, 3, 5, 10, 20]
    step_sizes = [1, 2, 5, 10]  

    output_folder = "AMZN_k_n_all"
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # --- Tạo A: các đoạn chuỗi của AMZN ---
    df_AMZN = df[df['Symbol'] == 'AMZN']
    df_AMZN = df_AMZN.sort_values('Date')
    values_AMZN = df_AMZN['Close'].values.astype(np.float32)
    scaler_AMZN = StandardScaler()
    scaled_AMZN = scaler_AMZN.fit_transform(values_AMZN.reshape(-1, 1)).flatten()
    
    # Tạo cửa sổ trượt và date_keys tương tự code2
    dates_AMZN = df_AMZN['Date'].values
    AMZN_date_keys = [
        f"AMZN_{pd.to_datetime(dates_AMZN[i+seq_len-1]).strftime('%Y%m%d')}"
        for i in range(len(scaled_AMZN)-window_size+1)
    ]
    
    # Cửa sổ quá khứ để phân tích
    AMZN_windows = np.array([scaled_AMZN[i:i+seq_len] for i in range(len(scaled_AMZN)-window_size+1)])
    num_AMZN = len(AMZN_windows)
    print(f"Number of AMZN windows: {num_AMZN}")

    # --- Tạo B: các đoạn chuỗi của tất cả cổ phiếu---
    df_industry = df.sort_values(['Symbol', 'Date'])
    symbol_list = df_industry['Symbol'].unique()

    b_windows = []
    b_meta = []  # Lưu lại symbol, date_key và dữ liệu tương lai
    
    for symbol in symbol_list:
        df_sym = df_industry[df_industry['Symbol'] == symbol]
        df_sym = df_sym.sort_values('Date')
        values_sym = df_sym['Close'].values.astype(np.float32)
        if len(values_sym) < window_size:
            continue
        
        dates_sym = df_sym['Date'].values
        
        scaler_sym = StandardScaler()
        scaled_sym = scaler_sym.fit_transform(values_sym.reshape(-1, 1)).flatten()
        
        # Tương tự code2: tạo date_keys và cửa sổ
        for i in range(len(scaled_sym) - window_size + 1):
            date_key = f"{symbol}_{pd.to_datetime(dates_sym[i+seq_len-1]).strftime('%Y%m%d')}"
            
            # Cửa sổ quá khứ để phân tích
            window = scaled_sym[i:i+seq_len]
            b_windows.append(window)
            
            # Dữ liệu giá trong tương lai (không chuẩn hóa)
            future_values = values_sym[i+seq_len:i+window_size]
            
            b_meta.append((symbol, date_key, future_values))
    
    b_windows = np.array(b_windows)
    num_b = len(b_windows)
    print(f"Number of windows in industry (B): {num_b}")

    # --- Tạo embedding ---
    try:
        # encoder = TCNEncoder(input_size=88, num_channels=[32, 64], kernel_size=3, dropout=0.1, emb_dim=64)
        encoder = TCNEncoder(input_size=486, num_channels=[32, 64], kernel_size=5, dropout=0.1, emb_dim=64)
        encoder.load_state_dict(torch.load("./Train_TCNT/results/tcn_all_encoder_best_9624.pt", map_location="cpu"))
        print(encoder)
        encoder.eval()

        with torch.no_grad():
            # Chuẩn bị tensor giả với 10 kênh
            batch_size = 100
            placeholder = torch.zeros((batch_size, 486, seq_len), dtype=torch.float32)
            
            # Xử lý AMZN windows theo batch
            AMZN_emb = []
            for i in range(0, num_AMZN, batch_size):
                current_batch_size = min(batch_size, num_AMZN - i)
                batch_placeholder = placeholder[:current_batch_size].clone()
                
                for j in range(current_batch_size):
                    batch_placeholder[j, 0, :] = torch.tensor(AMZN_windows[i+j], dtype=torch.float32)
                
                batch_emb = encoder(batch_placeholder).cpu().numpy()
                AMZN_emb.append(batch_emb)
                
            AMZN_emb = np.concatenate(AMZN_emb)
            print(f"Generated embeddings for {len(AMZN_emb)} AMZN windows")
            
            # Xử lý tất cả cổ phiếu theo batch
            b_emb = []
            for i in range(0, num_b, batch_size):
                current_batch_size = min(batch_size, num_b - i)
                batch_placeholder = placeholder[:current_batch_size].clone()
                
                for j in range(current_batch_size):
                    batch_placeholder[j, 0, :] = torch.tensor(b_windows[i+j], dtype=torch.float32)
                
                batch_emb = encoder(batch_placeholder).cpu().numpy()
                b_emb.append(batch_emb)
                
                if (i + batch_size) % 10000 == 0 or i + batch_size >= num_b:
                    print(f"Processing: {min(i + batch_size, num_b)}/{num_b} windows")
                
            b_emb = np.concatenate(b_emb)
            print(f"Generated embeddings for {len(b_emb)} industry windows")
    
        # --- Tìm kiếm tương đồng theo step_size ---
        for step_size in step_sizes:
            print(f"\n--- Step size = {step_size} ---")
            
            for top_k in top_k_list:
                print(f" Retrieval with top_k = {top_k}")
                all_future_values = []

                for offset in range(step_size):
                    group_indices = list(range(offset, num_AMZN, step_size))
                    
                    for i, global_i in enumerate(group_indices):
                        query = AMZN_emb[global_i]
                        dists = np.linalg.norm(b_emb - query, axis=1)
                        
                        # Lấy top_k+1 để loại bỏ trường hợp trùng với chính nó
                        topk_idx = np.argsort(dists)[:top_k+1]
                        
                        # Lọc ra các chuỗi không phải là chính nó
                        filtered_future_values = []
                        for idx in topk_idx:
                            sym, date_key, future_values = b_meta[idx]
                            if date_key != AMZN_date_keys[global_i]:  # Tránh trùng lặp với chính nó
                                filtered_future_values.append(future_values)
                                if len(filtered_future_values) == top_k:
                                    break
                        
                        # Thêm vào danh sách tổng
                        all_future_values.extend(filtered_future_values)
                
                # Chuyển thành tensor PyTorch và lưu - giống như code2
                if all_future_values:
                    reference_tensor = torch.tensor(all_future_values, dtype=torch.float32)
                    filename = f"AMZN_vs_all_{top_k}_{step_size}.pt"
                    filepath = os.path.join(output_folder, filename)
                    torch.save(reference_tensor, filepath)
                    print(f"✅ Saved {filepath} with shape {reference_tensor.shape}")
    
    except Exception as e:
        print(f"Error during embedding generation or similarity search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
