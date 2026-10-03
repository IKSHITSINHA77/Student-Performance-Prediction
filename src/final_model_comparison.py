import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "reports" / "final_model_results.csv"

RMSE_OUTPUT = BASE_DIR / "visualizations" / "final_model_rmse_comparison.png"
R2_OUTPUT = BASE_DIR / "visualizations" / "final_model_r2_comparison.png"

results = pd.read_csv(RESULTS_PATH)

# RMSE comparison
plt.figure(figsize=(10, 6))
plt.bar(results["Model"], results["RMSE"])
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")
plt.xlabel("Model")
plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RMSE_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

# R2 comparison
plt.figure(figsize=(10, 6))
plt.bar(results["Model"], results["R2"])
plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(R2_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"RMSE visualization saved to: {RMSE_OUTPUT}")
print(f"R2 visualization saved to: {R2_OUTPUT}")
