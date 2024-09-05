from .opbase import OpBase, verify_shape

class Plus(OpBase):
    """ Add two tensors together, must have same shape, broadcasting is not supported """
    def __init__(self, x1:OpBase, x2:OpBase):
        super().__init__(x1=x1, x2=x2)
        verify_shape(x1, x2, self.__repr__())

    @property
    def shape(self):
        return self.input_ops["x1"].shape

    def _forward(self):
        return self.input_ops["x1"].forward() + self.input_ops["x2"].forward()

    def _backward(self, input_op: OpBase):
        """ jabocian = identity matrix, grads pass straight through!"""
        self.grad = self.dloss_doutput()
        return self.grad

    def __repr__(self):
        return f"Plus: {self.shape}"
