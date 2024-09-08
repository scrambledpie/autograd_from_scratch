import numpy as np

from .opbase import OpBase


class Sin(OpBase):
    """ x -> """
    def __init__(self, x:OpBase):
        super().__init__(x=x)

    @property
    def shape(self):
        return self.input_ops["x"].shape

    def _forward(self):
        x = self.input_ops["x"].forward()
        self.cos_x = np.cos(x)
        self.sin_x = np.sin(x)
        return self.sin_x

    def _backward(self, input_op:OpBase):
        self.grad = self.dloss_doutput()
        return self.grad * self.cos_x

    def __repr__(self):
        return f"Sin {self.input_ops['x'].shape}"
