import numpy as np

from .ops import OpBase


class Reshape(OpBase):
    def __init__(self, x: OpBase, shape: tuple[int, int]):
        super().__init__(x=x)
        self._input_shape = x.shape
        self._shape_val = shape
    
    def _forward(self) -> np.ndarray:
        return self.input_ops["x"].forward().reshape(self.shape)
    
    def _backward(self, input_op: OpBase) -> np.ndarray:
        return self.dloss_doutput().reshape(self._input_shape)
    
    @property
    def shape(self):
        return self._shape_val
    
    def __repr__(self) -> str:
        return f"Reshape {self._input_shape} -> {self.shape}"
