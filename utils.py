import numpy as np
import gzip

data_dir = "mnist_data/"

file_names = {
    "train_image": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_image": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}


def load_image(fname):
    f = gzip.open(fname)
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28*28)


def load_label(fname):
    f = gzip.open(fname)
    data = np.frombuffer(f.read(), np.uint8, offset=8)
    data = data.flatten()
    return data


mnist_dataset = {}
for name, fname in file_names.items():
    if "image" in name:
        mnist_dataset[name] = load_image(data_dir + fname)
    elif "label" in name:
        mnist_dataset[name] = load_label(data_dir + fname)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print(mnist_dataset["train_label"][0])
    plt.imshow(mnist_dataset["train_image"][0].reshape(28, 28))
    plt.show()

