import numpy as np

from .opbase import OpBase


class Exp(OpBase):
    """ x -> """
    def __init__(self, x:OpBase):
        super().__init__(x=x)

    @property
    def shape(self):
        return self.input_ops["x"].shape

    def _forward(self):
        self.exp_x = np.exp(self.input_ops["x"].forward())
        return self.exp_x

    def _backward(self, input_op:OpBase):
        self.grad = self.dloss_doutput()
        return self.grad * self.exp_x

    def __repr__(self):
        return f"Exp {self.input_ops['x'].shape}"
