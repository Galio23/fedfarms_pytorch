import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('outputs/centralized/validation_metrics_by_farm.csv')
os.makedirs("outputs/visualization", exist_ok=True)

# List of soil properties in the dataset
soil_properties = ["C_gkg_filtered", "Clay_gkg_filtered"]

# Loop through each property and plot R2 and RMSE
for prop in soil_properties:
    # R2 Plot
    plt.figure(figsize=(14, 6))
    plt.bar(df["farm"], df[f"{prop}_R2"], color="skyblue")
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=90)
    plt.title(f"Farm-wise R² for {prop}")
    plt.xlabel("Farm")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(f"outputs/visualization/R2_{prop}centralized_farms.png")
    plt.close()

    # RMSE Plot
    plt.figure(figsize=(14, 6))
    plt.bar(df["farm"], df[f"{prop}_RMSE"], color="lightcoral")
    plt.xticks(rotation=90)
    plt.title(f"Farm-wise RMSE for {prop}")
    plt.xlabel("Farm")
    plt.ylabel("RMSE (g/kg)")
    plt.tight_layout()
    plt.savefig(f"outputs/visualization/RMSE_{prop}_centralized_farms.png")
    plt.close()
