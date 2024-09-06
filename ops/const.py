import numpy as np

from .opbase import OpBase, verify_shape


class Const(OpBase):
    """
    Constant: always return a predefined array (whose value can be manually
    changed at runtime). Serves as source nodes/variables in the tensor
    computational graph e.g. dataset, parameter matrices.
    """
    def __init__(self, val:np.ndarray):
        super().__init__()
        val = np.asarray(val)
        assert isinstance(val, np.ndarray), (
            f"Const requires np.ndarray: {type(val)}"
        )
        self.val = val

    def set_val(self, val: np.ndarray):
        verify_shape(self.val, val, "Const set_val:")
        self.val = val
        self._remove_cache()

    @property
    def shape(self):
        return self.val.shape

    def _forward(self) -> np.ndarray:
        return self.val

    def _backward(self, input_op=None):
        """ no input ops, nothing to compute! """
        raise NotImplementedError("Cannot backprop past a constant")

    def __repr__(self):
        return f"Const: {self.val.shape}"
