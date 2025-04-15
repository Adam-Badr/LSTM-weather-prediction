from TensorFlowModel import run_tensorflow_model
from PytorchModel import run_pytorch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


pred_tf, y_test, scalar = run_tensorflow_model()
pred_torch, _, _ = run_pytorch_model()  


y_true = scalar.inverse_transform(y_test)
pred_tf = scalar.inverse_transform(pred_tf)
pred_torch = scalar.inverse_transform(pred_torch)


feature_names = [
    "Temperature (2m)",
    "Relative Humidity (2m)",
    "Apparent Temperature",
    "Precipitation Probability",
    "Cloud Cover (High)",
    "Wind Direction (80m)"
]


def evaluate(true, pred, name):
    print(f"\n{name} Benchmark:")
    for i, feature in enumerate(feature_names):
        mae = mean_absolute_error(true[:, i], pred[:, i])
        mse = mean_squared_error(true[:, i], pred[:, i])
        print(f"{feature:30} MAE: {mae:.4f}, MSE: {mse:.4f}")


evaluate(y_true, pred_tf, "TensorFlow")
evaluate(y_true, pred_torch, "PyTorch")


for i in range(6):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:, i], label="Actual", linewidth=1.5)
    plt.plot(pred_tf[:, i], label="TensorFlow", linestyle="--")
    plt.plot(pred_torch[:, i], label="PyTorch", linestyle=":")
    plt.title(f"{feature_names[i]} - Prediction Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()