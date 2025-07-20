import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

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

def main():
    csv_path = "AMZN_industry.csv"
    seq_len = 96
    pred_len = 24
    top_k_list = [1, 3, 5, 10, 20]
    step_sizes = [1, 2, 5, 10]

    output_folder = "AMZN_k_n_only"
    os.makedirs(output_folder, exist_ok=True)

    # Đọc và xử lý dữ liệu
    df = pd.read_csv(csv_path)
    df = df[df['Symbol'] == 'AMZN']
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    values = df['Close'].values.astype(np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    total_days = len(scaled)
    valid_samples = total_days - seq_len - pred_len + 1
    print(f"Total days: {total_days}")
    print(f"Valid samples: {valid_samples}")

    # Tạo history windows (96 ngày) và future windows (20 ngày) - GIỮ NGUYÊN GIÁ GỐC
    history_windows = []
    future_windows_original = []  # Lưu giá gốc, không chuẩn hóa
    
    for i in range(valid_samples):
        history = scaled[i:i+seq_len]  # 96 ngày (chuẩn hóa cho encoder)
        future_original = values[i+seq_len:i+seq_len+pred_len]  # 20 ngày GIÁ GỐC
        history_windows.append(history)
        future_windows_original.append(future_original)
    
    history_windows = np.array(history_windows)
    future_windows_original = np.array(future_windows_original)

    # Chuyển thành tensor cho encoder
    xH_tensor = torch.tensor(history_windows, dtype=torch.float32).unsqueeze(1)

    # Tải mô hình TCNEncoder
    encoder = TCNEncoder(
        input_size=1,
        num_channels=[32, 64],
        kernel_size=5,
        dropout=0.2,
        emb_dim=32
    )
    encoder.load_state_dict(torch.load("./Train_TCNT/results/tcn_only_encode_best_AMZN_9624.pt", map_location="cpu"))
    encoder.eval()

    # Tạo embedding
    with torch.no_grad():
        xH_embeddings = encoder(xH_tensor).cpu().numpy()

    # Thử nghiệm với các step_size
    for step_size in step_sizes:
        print(f"\n--- Step size = {step_size} ---")
        reference_indices = [None] * valid_samples

        # Tìm kiếm tương đồng trong từng nhóm theo step_size
        for offset in range(step_size):
            group_indices = list(range(offset, valid_samples, step_size))
            group_embeddings = xH_embeddings[group_indices]

            for i, global_i in enumerate(group_indices):
                query = group_embeddings[i]
                dists = np.linalg.norm(group_embeddings - query, axis=1)
                dists[i] = np.inf
                topk_local = np.argsort(dists)[:max(top_k_list)]
                reference_indices[global_i] = [group_indices[j] for j in topk_local]

        # Lưu kết quả cho từng top_k
        for top_k in top_k_list:
            all_future_values = []
            
            for ref in reference_indices:
                if ref is not None:
                    # Lấy top_k indices
                    top_k_indices = ref[:top_k]
                    # Lấy 20 ngày tiếp theo GIÁ GỐC của các chuỗi tương đồng
                    top_k_future_values = future_windows_original[top_k_indices]  # (top_k, 20)
                    # Flatten để có shape (top_k * 20,)
                    for future_seq in top_k_future_values:
                        all_future_values.append(future_seq)
            
            # Chuyển thành tensor với shape (valid_samples * top_k, 20)
            reference_tensor = torch.tensor(all_future_values, dtype=torch.float32)
            
            filename = f"AMZN_vs_only_{top_k}_{step_size}.pt"
            filepath = os.path.join(output_folder, filename)
            torch.save(reference_tensor, filepath)
            print(f" Saved to {filepath} with shape: {reference_tensor.shape}")

if __name__ == "__main__":
    main()
