import numpy as np
from utils import mnist_dataset

X_train = mnist_dataset["train_image"]
y_train = mnist_dataset["train_label"]
X_test = mnist_dataset["test_image"]
y_test = mnist_dataset["test_label"]

X_train = X_train[:10000].astype(np.float16)
y_train = y_train[:10000].astype(np.float16)
epochs = 20
