import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('metrics_farm/validation_metrics_by_farm.csv')
# Plot farm-wise R²
plt.figure(figsize=(14, 6))
plt.bar(df["farm"], df["R2"], color="skyblue")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xticks(rotation=90)
plt.title("Farm-wise R² Scores")
plt.xlabel("Farm")
plt.ylabel("R²")
plt.tight_layout()
plt.savefig("visualization/R2_plot.png")
#plt.show()

# Plot farm-wise RMSE
plt.figure(figsize=(14, 6))
plt.bar(df["farm"], df["RMSE"], color="lightcoral")
plt.xticks(rotation=90)
plt.title("Farm-wise RMSE")
plt.xlabel("Farm")
plt.ylabel("RMSE (g/kg)")
plt.tight_layout()
plt.savefig("visualization/RMSE_plot.png")
#plt.show()
