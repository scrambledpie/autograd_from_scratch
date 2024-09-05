from .opbase import OpBase


class Transpose(OpBase):
    """ x -> """
    def __init__(self, x:OpBase):
        super().__init__(x=x)

    @property
    def shape(self):
        return (self.input_ops["x"].shape[1], self.input_ops["x"].shape[0])

    def _forward(self):
        return self.input_ops["x"].forward().transpose()

    def _backward(self, input_op:OpBase):
        self.grad = self.dloss_doutput().transpose()
        return self.grad

    def __repr__(self):
        return f"Transpose {self.input_ops['x'].shape} -> {self.shape}"
