from .opbase import OpBase, verify_shape


class HProd(OpBase):
    """ x -> """
    def __init__(self, x:OpBase, y:OpBase):
        super().__init__(x=x, y=y)
        verify_shape(x, y, "HProd: ")

    @property
    def shape(self):
        return self.input_ops["x"].shape

    def _forward(self):
        self.x = self.input_ops["x"].forward()
        self.y = self.input_ops["y"].forward()
        return self.x * self.y

    def _backward(self, input_op:OpBase):
        self.grad = self.dloss_doutput()

        if id(input_op) == id(self.input_ops["x"]):
            return self.grad * self.y
        elif id(input_op) == id(self.input_ops["y"]):
            return self.grad * self.x

    def __repr__(self):
        return f"Hprod {self.input_ops['x'].shape}"
