import numpy as np

from .opbase import OpBase

class SumReduce(OpBase):
    """ Sum all the elements in a tensor and output a scalar """
    def __init__(self, x:OpBase):
        super().__init__(x=x)

    @property
    def shape(self):
        return (1, 1)

    def _forward(self):
        data = self.input_ops["x"].forward()
        return np.sum(data).reshape((1,1))

    def _backward(self, input_op: OpBase):
        """ Jacobian = all ones matrix, everyone contributes equally! """
        return self.dloss_doutput() * np.ones(self.input_ops["x"].shape)

    def __repr__(self):
        return f"SumReduce {self.input_ops['x'].shape} -> (1,1)"
