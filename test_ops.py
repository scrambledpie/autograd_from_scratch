import unittest
from typing import Type

import numpy as np

from ops import (
    CELoss,
    Clip,
    Const,
    Cos,
    Exp,
    Loss,
    L2Distance,
    Matmul,
    MaxReduce,
    OpBase,
    Plus,
    Reshape,
    ScalarMultiply,
    Sin,
    SumReduce,
    Transpose,
)


def grad_fin_diff(f:callable, x:np.array, h:float=1e-4) -> np.array:
    """
    Use finite differnces to get the gradient of a function with repsect to
    every input element.
    """
    df_dx = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(x[i])):
            x_ij_old = float(x[i, j])

            x[i, j] = x_ij_old + h
            y_up = f(x)

            x[i, j] = x_ij_old - h
            y_down = f(x)

            df_dx[i, j] = (y_up - y_down ) / (2 * h)

            x[i,j] = x_ij_old

    return df_dx


def check_tensors_close(
        x:np.ndarray,
        x_true: np.ndarray,
        rtol=0.01
    ) -> list[str]:
    """ compare arrays element-wise and return mismatched elements """
    assert x.shape == x_true.shape, f"{x.shape} {x_true.shape} not equal"
    errors = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            delta = (x[i, j] - x_true[i, j])
            delta = delta / (x_true[i, j] + 1e-9)

            if delta > rtol:
                errors.append([i, j, delta])
    return errors


class TestOps(unittest.TestCase):
    @staticmethod
    def verify_1d_op(
        op_cls: Type[OpBase],
        x_shape:tuple[int, int]=(15, 4),
        **op_cls_kwargs,
    ):
        """
        Check the gradients of an operator that takes a single input tensor.
        The operator is wrapped in a ReduceSum and the gradient of this summed
        outpout is checked.
        """

        def val_and_grad(x:np.ndarray) -> tuple[float, np.ndarray]:
            x = Const(x)
            y = Loss(SumReduce(op_cls(x, **op_cls_kwargs)))
            val = y.forward()
            grad = x.dloss_doutput()
            return val[0, 0], grad

        np.random.seed(42)
        x_test = np.random.uniform(size=x_shape)

        grad_fd = grad_fin_diff(lambda x: val_and_grad(x)[0], x_test)
        grad_backprop = val_and_grad(x_test)[1]

        errors = check_tensors_close(grad_backprop, grad_fd)
        if errors:
            raise Exception("bad gradients:", errors)

    @staticmethod
    def verify_2d_op(op_cls: Type[OpBase], x_shape=(15, 10), y_shape=(10, 15)):

        def val_and_grads(
            x:np.ndarray,
            y:np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            x = Const(x)
            y = Const(y)
            z = Loss(SumReduce(op_cls(x, y)))
            val = z.forward()[0, 0]
            grad_x = x.dloss_doutput()
            grad_y = y.dloss_doutput()
            return val, grad_x, grad_y

        np.random.seed(42)
        x_test = np.random.uniform(size=x_shape)
        y_test = np.random.uniform(size=y_shape)
        grad_x = grad_fin_diff(lambda x: val_and_grads(x, y_test)[0], x_test)
        grad_y = grad_fin_diff(lambda y: val_and_grads(x_test, y)[0], y_test)
        _, grad_x_theory, grad_y_theory = val_and_grads(x_test, y_test)

        errors_y = check_tensors_close(grad_y_theory, grad_y)
        if errors_y:
            raise Exception("Y bad gradients:", errors_y)

        errors_x = check_tensors_close(grad_x_theory, grad_x)
        if errors_x:
            raise Exception("X bad gradients:", errors_x)


    def test_clip(self):
        self.verify_1d_op(Clip)

    def test_sumreduce(self):
        self.verify_1d_op(SumReduce)

    def test_maxreduce(self):
        self.verify_1d_op(MaxReduce)

    def test_transpose(self):
        self.verify_1d_op(Transpose)

    def test_scalarmult(self):
        self.verify_1d_op(ScalarMultiply)

    def test_exp(self):
        self.verify_1d_op(Exp)

    def test_sin(self):
        self.verify_1d_op(Sin)

    def test_cos(self):
        self.verify_1d_op(Cos)

    def test_Reshape(self):
        self.verify_1d_op(Reshape, x_shape=(15, 4), shape=(5, 12))

    def test_plus(self):
        self.verify_2d_op(Plus, x_shape=(15, 4), y_shape=(15, 4))

    def test_matmul(self):
        self.verify_2d_op(Matmul, x_shape=(15, 4), y_shape=(4, 15))

    def test_L2Dist(self):
        self.verify_2d_op(L2Distance, x_shape=(15, 4), y_shape=(15, 4))

    def test_CEloss(self):
        self.verify_2d_op(CELoss, x_shape=(15, 4), y_shape=(15, 4))




if __name__=="__main__":
    unittest.main()

