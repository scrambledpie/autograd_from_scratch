from pathlib import Path
import pickle as pkl

import numpy as np

MNIST_FILE = Path(__file__).parent / "mnist.pkl"

if not MNIST_FILE.exists():
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    y_train_oh = np.zeros((60000, 10))
    y_train_oh[np.arange(60000), y_train] = 1

    y_test_oh = np.zeros((10000, 10))
    y_test_oh[np.arange(10000), y_test] = 1

    data = [x_train, y_train_oh, x_test, y_test_oh]

    with open(MNIST_FILE, "wb") as f:
        pkl.dump(data, f)


with open(MNIST_FILE, "rb") as f:
    MNIST_DATASET = pkl.load(f)
    print(f"Loaded MNIST: {MNIST_FILE}")





