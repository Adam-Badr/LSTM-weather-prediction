from tensorflow_model import run_tensorflow_model
from pytorch_model import run_pytorch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

#Run both files
pred_tf, y_tf, scalar_tf = run_tensorflow_model()
pred_torch, y_torch, scalar_torch = run_pytorch_model()

#Inverse everything
true_tf = scalar_tf.inverse_transform(y_tf)
pred_tf = scalar_tf.inverse_transform(pred_tf)
pred_torch = scalar_torch.inverse_transform(pred_torch)


def evaluate(true, pred, name):
    print(f"\n {name} Benchmark:")
    print("MAE:", mean_absolute_error(true, pred))
    print("MSE:", mean_squared_error(true, pred))

evaluate(true_tf, pred_tf, "TensorFlow")
evaluate(true_tf, pred_torch, "PyTorch")


plt.plot(true_tf[:, 0], label="Actual Temp")
plt.plot(pred_tf[:, 0], label="TF Prediction")
plt.plot(pred_torch[:, 0], label="Torch Prediction")
plt.title("Temperature Predictions")
plt.legend()
plt.tight_layout()
plt.show()