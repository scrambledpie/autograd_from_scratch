import time

import numpy as np

from ops import Clip, Const, Matmul, CELoss, SumReduce, Loss, ScalarMultiply
from mnist_dataset import MNIST_DATASET

np.random.seed(0)

X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = MNIST_DATASET

# Flatten images from (60000, 28, 27) -> (60000, 784)
X_TRAIN = X_TRAIN.reshape(60000, 784) / 255
X_TEST = X_TEST.reshape(10000, 784) / 255

# The computaional graph is designed to take 60,000 inputs at once. Expand the
# test set to the same size as the training (wasteful, but easy!)
X_TEST = np.tile(X_TEST, (6, 1))
Y_TEST = np.tile(Y_TEST, (6, 1))

# These make measuring accuracy easier
Y_TRAIN_IDX = np.argmax(Y_TRAIN, axis=1)
Y_TEST_IDX = np.argmax(Y_TEST, axis=1)


class MLP:
    """ Multi Layer Perceptron to train and Predict MNIST digits. """
    def __init__(self, learning_rate:float=0.00001):

        self.learning_rate = learning_rate

        # Define the computational graph
        # Inputs 1: Data
        x = Const(X_TRAIN)  # (60000, 784)
        y = Const(Y_TRAIN)  # (60000, 10)

        # Inputs 2: Params for 3 fully connected layers 784 -> 500 -> 450 -> 10
        w1 = Const(np.random.normal(size=(784, 500)) * (1/np.sqrt(784)))
        w2 = Const(np.random.normal(size=(500, 450)) * (1/np.sqrt(500)))
        w3 = Const(np.random.normal(size=(450, 10)) * (1/np.sqrt(450)))

        # Computational graph from inputs to logits and loss
        x1 = Clip(Matmul(x, w1), min=0, max=1e9)  # (60000, 500)
        x2 = Clip(Matmul(x1, w2), min=0, max=1e9) # (60000, 450)
        logits = Matmul(x2, w3)  # (60000, 10)
        loss_per_item = CELoss(logits, y)  # (60000, 1)
        loss_sum =SumReduce(loss_per_item) # (1, 1)
        loss_mean = ScalarMultiply(loss_sum, 1/x.shape[0])  # (1 ,1)
        loss = Loss(loss_mean)  # (1, 1)

        # store these refernces for direct access
        self.x = x
        self.y = y
        self.params = [w1, w2, w3]
        self.logits = logits
        self.loss = loss

        print("\n", self.loss.print_inputs(), "\n\n")

    def test_accuracy(self) -> tuple[float, float]:
        """
        Compute train set and test set prediction accuracy
        """
        self.x.set_val(X_TEST)
        logits_np = self.logits.forward()
        y_pred_idx = np.argmax(logits_np, axis=1)
        test_acc = np.mean(y_pred_idx == Y_TEST_IDX)

        self.x.set_val(X_TRAIN)
        logits_np = self.logits.forward()
        y_pred_idx = np.argmax(logits_np, axis=1)
        train_acc = np.mean(y_pred_idx == Y_TRAIN_IDX)

        return train_acc, test_acc

    def update_params(self) -> float:
        """
        Compute gradients, update weights with gradient descent, return loss
        """
        loss_value = self.loss.forward()[0, 0]
        for w in self.params:
            grad_w = w.dloss_doutput()
            w.set_val(w.val - self.learning_rate * grad_w)
        return loss_value

    def train(self):
        """ name says it all really..... """
        print("0: train_acc, test_acc before: ", self.test_accuracy())

        for i in range(1000):

            tick = time.time()
            loss_value = self.update_params()

            # Compute metrics and Store metrics and print them out
            train_acc, test_acc = self.test_accuracy()
            print(
                f"{i}: loss={loss_value:3f}"
                f"(prob(y_true)={np.exp(-loss_value):4f}) "
                f"train_acc={train_acc:2f}, test_ac={test_acc:2f}, "
                f"{(time.time() - tick):.3f} seconds"
            )


model = MLP()
model.train()
