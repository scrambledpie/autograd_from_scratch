import numpy as np

from .ops import OpBase


class Matmul(OpBase):
    """ Take two matrices and matrix multiuply (n, k) X (k, m) -> (n, m) """
    def __init__(self, x1:OpBase, x2:OpBase):
        super().__init__(x1=x1, x2=x2)

        assert x1.shape[1] == x2.shape[0], (
            f"Matmul incompatible shapes {x1.shape} {x2.shape}"
        )
        self.shape_val = (x1.shape[0], x2.shape[1])

    def _forward(self):
        self.x1_val = self.input_ops["x1"].forward()
        self.x2_val = self.input_ops["x2"].forward()
        return np.matmul(self.x1_val, self.x2_val)

    def _backward(self, input_op: OpBase) -> np.ndarray:
        grad_out = self.dloss_doutput()

        if id(input_op) == id(self.input_ops["x1"]):
            return np.matmul(grad_out, self.x2_val.transpose())

        elif id(input_op) == id(self.input_ops["x2"]):
            return np.matmul(self.x1_val.transpose(), grad_out)

        else:
            raise Exception(f"Unknown input op {input_op}")

    @property
    def shape(self):
        return self.shape_val

    def __repr__(self):
        return f"Matmul {self.shape}"
