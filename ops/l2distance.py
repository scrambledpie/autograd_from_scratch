from .ops import OpBase, verify_shape

class L2Distance(OpBase):
    """ element-wise squared differences, can be used for squared loss """
    def __init__(self, x1:OpBase, x2:OpBase):
        super().__init__(x1=x1, x2=x2)
        verify_shape(x1, x2, self.__repr__())

    @property
    def shape(self):
        return self.input_ops["x1"].shape

    def _forward(self):
        self.diff = self.input_ops["x1"].forward() - self.input_ops["x2"].forward()
        return 0.5 * (self.diff)**2

    def _backward(self, input_op: OpBase):
        """ jacobian: doutput/dx1 = (x1 - x2), doutput/dx2 = (x2 - x1) """
        self.grad = self.diff * self.dloss_doutput()

        if id(input_op) == id(self.input_ops["x1"]):
            return self.grad
        elif id(input_op) == id(self.input_ops["x2"]):
            return -self.grad
        else:
            raise Exception(f"Unknown Parent Op {input_op}")

    def __repr__(self):
        return f"L2 Loss {self.shape}"
