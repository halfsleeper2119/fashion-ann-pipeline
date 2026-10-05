from pathlib import Path
import numpy as np
from tensorflow import keras

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
np.savez_compressed(out / "fashion_mnist.npz",
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)
print("Saved raw data to", out)