import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.stdout.reconfigure(encoding="utf-8")

# Load Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "household_energy_preprocessed.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Energy_Consumption_kWh"])
y = df["Energy_Consumption_kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAINING (Logging langsung, pakai active run dari project)
# =========================

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Logging
mlflow.log_param("model_type", "RandomForestRegressor")
mlflow.log_param("n_estimators", 150)

mlflow.log_metric("mse", mse)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2_score", r2)

# Log model
mlflow.sklearn.log_model(model, artifact_path="model")

# Artifact 1: Predictions
pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})
pred_df.to_csv("predictions.csv", index=False)
mlflow.log_artifact("predictions.csv")

# Artifact 2: Feature Importance
plt.figure(figsize=(8, 6))
plt.bar(X.columns, model.feature_importances_)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

mlflow.log_artifact("feature_importance.png")

# Output
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print("Run successfully logged via MLflow Project")