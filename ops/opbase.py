from abc import ABC, abstractmethod

import numpy as np


def verify_shape(x1:np.ndarray, x2: np.ndarray, msg:str="") -> None:
    """ check shapes of input ops/arrays and raise erorrs of mismatches """
    s1 = x1.shape
    s2 = x2.shape
    assert len(s1) == 2, f"{msg}: arrays must be 2D, {len(s1)}"
    assert len(s2) == 2, f"{msg}: arrays must be 2D, {len(s2)}"
    assert all([s1i==s2i for s1i, s2i in zip(s1, s2)]), (
        f"{msg} shape mismatch {s1} {s2}"
    )


class OpBase(ABC):
    """
    Base class for all operators, handles input/output linking and caching of
    results for forward/backward passes.
    """
    def __init__(self, **kwargs):
        # kwargs: a dict of input names and their ops
        # this node consumes tensors from these input_ops
        self.input_ops = {}
        self._input_op_name_from_id = {}
        for name, op in kwargs.items():
            assert issubclass(type(op), OpBase), (
                f"{self}: invalid input type {type(op)}, must be"
                " an operator (subclass of 'Op')"
            )
            self.input_ops[name] = op
            op.output_ops.append(self)
            self._input_op_name_from_id[id(op)] = name

        # A list of downstream ops that consume this ops output (to be populated
        # as those output ops are instantiated)
        self.output_ops = []

        # forward/backward caching
        self._forward_cached: np.ndarray = None  # one tensor
        self._backward_cached: dict[str, np.ndarray] = {}  # one per input_op

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """ the output shape of the operator, output of self.forward() """
        pass

    @abstractmethod
    def _forward(self) -> np.ndarray:
        """ All computation return numpy array of self.shape """
        pass

    def _remove_cache(self) -> None:
        """ remove cached output from this Op and all downstream Ops """
        self._forward_cached = None
        self._backward_cached = {}
        for output_op in self.output_ops:
            output_op._remove_cache()

    def forward(self) -> np.ndarray:
        """ compute and cache outputs and check shapes """
        # for intermediate computational nodes, we need to check the version of
        # all input tensors.
        if len(self.input_ops) == 0:
            # no caching for comp graph source nodes (Const)
            return self._forward()

        if self._forward_cached is None:
            self._forward_cached = self._forward()
            verify_shape(self, self._forward_cached, f"{self} forward: ")

        return self._forward_cached

    def dloss_doutput(self) -> np.ndarray:
        """
        Compute dLoss/doutput where output = self.forward(), compute the loss
        gradient with respect to each element of this ops output tensor.
        dLoss/doutput = sum dloss_dinput gathered from self.output_ops.

        Ensure dLoss/doutput shape matches self.forward() shape.
        """
        grad_out = np.zeros(shape=self.shape)
        for output_op in self.output_ops:
            grad_out_i = output_op.dloss_dinput(input_op = self)
            verify_shape(grad_out_i, self, f"{self}: received grad {output_op}")
            grad_out += grad_out_i
        return grad_out

    @abstractmethod
    def _backward(self, input_op) -> np.ndarray:
        """
        - fetch dLoss/doutput (self.shape)
        - compute dLoss/dinput_op = dLoss/doutput X doutput/dinput_op which must
            have shape (input_op.shape)

        where doutput/dinput_op is the jacobian matrix, however the full matmul
        is rarely required and the output product can be computed cheaply.
        """
        pass

    def dloss_dinput(self, input_op) -> np.ndarray:
        """
        Compute dLoss/dinput_op where input_op is a single input tensor.

        Note:the gradient returned is only how the input affects loss through
        this op. The same input tensor may be used in other ops and affect the
        loss in other ways, this is only a local partial derivitive.

        E.g. let this op represent f(x) and we have Loss(x) = f(x) + g(x) then
        the full gradient w.r.t. x is given by
                dLoss/dx = dLoss/df * df/dx + dLoss/dg * dg/dx

        However, this op only returns dLoss/df * df/dx.
        """
        try:
            input_name = self._input_op_name_from_id[id(input_op)]
        except KeyError:
            raise KeyError(f"unexpected input_op {input_op}")

        if input_name not in self._backward_cached:
            self._backward_cached[input_name] = self._backward(input_op)
            verify_shape(
                self._backward_cached[input_name],
                input_op,
                f"{self}: grad shape"
            )

        return self._backward_cached[input_name]

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def print_inputs(self) -> str:
        """
        Create a string representation of the computational graph up until self
        """
        my_str = f"\t{self}"
        for op in self.input_ops.values():
            my_str = op.print_inputs() + "\n" + my_str
        return my_str

