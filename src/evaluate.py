import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras

d = np.load("data/processed/fashion_mnist.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = model.predict(d["x_test"], verbose=0).argmax(axis=1)

ConfusionMatrixDisplay.from_predictions(d["y_test"], pred)
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

json.dump({"test_loss": float(loss), "test_accuracy": float(acc)},
          open("metrics.json", "w"), indent=2)
print(f"loss={loss:.4f} accuracy={acc:.4f}")