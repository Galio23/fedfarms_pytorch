import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from kennard_stone import train_test_split
import numpy as np
from model_pytorch_multioutput import create_model
import os
import joblib

# Load dataset
df = pd.read_csv("data/farms_sentinel.csv")  # Replace with actual path if needed
y_scaler = joblib.load("code/models/scaler_y_2cols.pkl")
# Initialize metrics list
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = (10, 1)  # 10 timesteps (X1–X10), 1 channel
output_shape = 2

model = create_model(input_shape, output_shape)
model.load_state_dict(torch.load("code/models/best_model_notest.pth", map_location=device))
model.to(device)    
model.eval()

def main():
    def evaluate_model(model, X_val, y_val, y_scaler):

        X_tensor = (
            torch.tensor(X_val.values, dtype=torch.float32)
            .unsqueeze(-1)
            .to(device)
        )
        with torch.no_grad():
            preds_scaled = model(X_tensor)    # now both input&model are on the same device
            preds_scaled = preds_scaled.cpu().numpy()
    
        
        
        y_pred_orig = y_scaler.inverse_transform(preds_scaled)  
        y_true_orig = y_val.values                              
        
        # 4) Compute metrics per property
        out_metrics = {}
        for i, prop in enumerate(y_val.columns):
            y_t = y_true_orig[:, i]
            y_p = y_pred_orig[:, i]
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            rpiq = (np.percentile(y_t, 75) - np.percentile(y_t, 25)) / rmse if rmse != 0 else np.nan
            out_metrics[prop] = {
                'R2':   r2_score(y_t, y_p),
                'RMSE': rmse,
                'RPIQ': rpiq,
            }
        return out_metrics


        
        # Loop through each farm
    for farm_id in df['farm'].unique():
        farm_df = df[df['farm'] == farm_id]
        X = farm_df[[f'X{i}' for i in range(1,11)]]
        y = farm_df[['Clay_gkg_filtered','C_gkg_filtered']]  # must match scaler order

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
        metrics = evaluate_model(model, X_val, y_val, y_scaler)
        results.append({
            'farm': farm_id,
            **{f"{prop}_{m}": val for prop, mets in metrics.items() for m, val in mets.items()},
            'Samples': len(y_val),
        })
        print(f"Farm {farm_id}: {metrics}")


        

    metrics_df = pd.DataFrame(results)
    os.makedirs("outputs/centralized", exist_ok=True)
    metrics_df.to_csv("outputs/centralized/validation_metrics_by_farm.csv", index=False)

    print("✅ Metrics saved to validation_metrics_by_farm.csv")

if __name__ == "__main__":
    main()