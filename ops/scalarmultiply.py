from .ops import OpBase


class ScalarMultiply(OpBase):
    """ x -> a * x """
    def __init__(self, x:OpBase, mult:float=24.8):
        super().__init__(x=x)
        self.mult = mult
        self.inv_mult = 1 / mult

    @property
    def shape(self):
        return self.input_ops["x"].shape

    def _forward(self):
        return self.mult * self.input_ops["x"].forward()

    def _backward(self, input_op:OpBase):
        return self.dloss_doutput() * self.inv_mult

    def __repr__(self):
        return f"ScalarMult {self.mult} {self.shape}"
