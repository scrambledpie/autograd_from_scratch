import numpy as np

from .opbase import OpBase


class Loss(OpBase):
    """
    Final op to initiate the backward pass, start with a tensor of a single
    scalar 1.
    """
    def __init__(self, x:OpBase):
        assert x.shape[0] == 1, f"incorrect final shape {x.shape} != (1,1)"
        assert x.shape[1] == 1, f"incorrect final shape {x.shape} != (1,1)"
        assert len(x.shape) == 2

        super().__init__(x=x)

    def _forward(self):
        return self.input_ops["x"].forward()

    def _backward(self, input_op: OpBase):
        return 1.0 * np.ones((1, 1))

    @property
    def shape(self):
        return (1, 1)

    def __repr__(self):
        return "Loss"
