from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

raw = np.load("data/raw/fashion_mnist.npz")

x_train = raw["x_train"].astype("float32") / 255.0
x_test  = raw["x_test"].astype("float32") / 255.0
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_test  = (x_test - mean) / std

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, raw["y_train"],
    test_size=params["test_size"], random_state=params["seed"],
    stratify=raw["y_train"])

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "fashion_mnist.npz",
                    x_train=x_tr, y_train=y_tr,
                    x_val=x_val, y_val=y_val,
                    x_test=x_test, y_test=raw["y_test"])
print("Saved processed data to", out)