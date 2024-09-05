import numpy as np

from .opbase import OpBase


class MaxReduce(OpBase):
    """ Get max of all the elements in a tensor and output a scalar """
    def __init__(self, x:OpBase):
        super().__init__(x=x)

    @property
    def shape(self):
        return (1, 1)

    def _forward(self):
        x = self.input_ops["x"].forward()
        self.best_x = np.argmax(x.reshape(-1))
        return np.max(x).reshape((1,1))

    def _backward(self, input_op: OpBase) -> np.ndarray:
        """ Jacobian = all zeros except 1 at location of max element """

        grad_out = self.dloss_doutput()

        mask = np.zeros(self.input_ops["x"].shape).reshape(-1)
        mask[self.best_x] = 1
        mask = mask.reshape(self.input_ops["x"].shape)

        self.grad = grad_out * mask
        return self.grad

    def __repr__(self):
        return f"MaxReduce {self.input_ops['x'].shape} -> (1,1)"
