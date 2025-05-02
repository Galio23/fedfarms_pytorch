import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from kennard_stone import train_test_split
import numpy as np
from pytorch.model_pytorch import create_model
import os
import joblib

# Load dataset
df = pd.read_csv("data/farms_sentinel.csv")  # Replace with actual path if needed
y_scaler = joblib.load("models/y_scaler.pkl")
# Initialize metrics list
results = []

input_shape = (10, 1)  # 10 timesteps (X1–X10), 1 channel
output_shape = 1

model = create_model(input_shape, output_shape)
model.load_state_dict(torch.load("models/centralized_general_model.pth"))
model.eval()

def evaluate_model(model, X_val, y_val, y_scaler):
    X_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(-1)  # shape: (N, 10, 1)
    
    # Scale y_val
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).squeeze()

    with torch.no_grad():
        preds_scaled = model(X_tensor).squeeze().numpy()

    # Inverse transform to get original values
    y_pred_orig = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
    y_val_orig = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).squeeze()

    r2 = r2_score(y_val_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
    rpiq = (np.percentile(y_val_orig, 75) - np.percentile(y_val_orig, 25)) / rmse if rmse != 0 else float("inf")

    return r2, rmse, rpiq

    
    # Loop through each farm
for farm_id in sorted(df['farm'].unique(), key=lambda x: int(x.split("_")[1])):
    farm_df = df[df['farm'] == farm_id].reset_index(drop=True)
    
    X = farm_df[[f'X{i}' for i in range(1, 11)]]
    y = farm_df['Clay_gkg_filtered']
    
    # Use Kennard-Stone to split
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8,random_state= 42)
    print(f'Evaluating farm {farm_id}..')
    # Evaluate
    r2, rmse, rpiq = evaluate_model(model, X_val, y_val,y_scaler)
    results.append({
        'farm': farm_id,
        'R2': r2,
        'RMSE': rmse,
        'RPIQ': rpiq,
        'Samples': len(y_val)
    
    })
    

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("metrics_farm/validation_metrics_by_farm.csv", index=False)

print("✅ Metrics saved to validation_metrics_by_farm.csv")