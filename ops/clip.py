import numpy as np

from .ops import OpBase

class Clip(OpBase):
    """ Clip tensor values to min/max range (e.g. like RELU activation function) """
    def __init__(self, x: OpBase, min:float=0, max:float=1e9):
        super().__init__(x=x)
        self.min = min
        self.max = max

    @property
    def shape(self):
        return self.input_ops["x"].shape

    def _forward(self):
        x = self.input_ops["x"].forward()
        self.mask = 1.0 * ((self.min < x) & (x < self.max))
        return np.clip(x, a_min=self.min, a_max=self.max)

    def __repr__(self):
        return f"Clip {self.shape} {self.min}, {self.max}"

    def _backward(self, input_op: OpBase):
        """ jacobian = Identity matrix with 0s on the diagonal for clipped values """
        return self.mask * self.dloss_doutput()
