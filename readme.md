# RATD Stock Forecasting Pipeline
## Project Structure
- `Train_TCN/` - TCN model training scripts and results
- `Retrival/` - Retrieval scripts and outputs
- `Diffusion/` - Diffusion model scripts, configs, and outputs
- `dataset/` - Datasets for training and evaluation

The workflow is organized into three main stages:

## 1. Train TCN Embedding Models
Train Temporal Convolutional Network (TCN) models to generate embeddings for stocks.

**Run training:**
```bash
cd Train_TCN
python train_tcn_all.py         # For all stocks
python train_tcn_industry.py    # For industry-specific
python train_tcn_only.py        # For single stock
```

- Model weights and embeddings will be saved in `Train_TCN/results/`.

## 2. Retrieval Stage
Use the trained TCN models to generate retrieval features for downstream tasks.

**Run retrieval:**
```bash
cd Retrival
python retri_all.py         # For all stocks
python retri_industry.py    # For industry-specific
python retri_only.py        # For single stock
```
- Retrieval outputs will be saved in the corresponding folders (e.g., `Retrival/AMZN_k_n_all/`).

## 3. Diffusion Stage
Feed the retrieval outputs into the diffusion models for final forecasting.

**Run diffusion forecasting:**
```bash
cd Diffusion
python exe_stock_forecasting_all.py         # For all stocks
python exe_stock_forecasting_industry.py    # For industry-specific
python exe_stock_forecasting_only.py        # For single stock
```



