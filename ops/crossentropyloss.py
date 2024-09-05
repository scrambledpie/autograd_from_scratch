import numpy as np

from .ops import OpBase, verify_shape


class CELoss(OpBase):
    """
    Cross-entropy loss, consuming logits and one-hot y labels.
    (batch_size, num_classes)  (batch_size, num_classes) -> (batch_size, 1)
    """
    def __init__(self, logits:OpBase, y: OpBase):
        verify_shape(logits, y, "CE loss")
        self.bs, self.classes = logits.shape
        super().__init__(logits=logits, y=y)

    @property
    def shape(self):
        # (batch size, 1)
        return (self.bs, 1)

    def _forward(self):
        # (bs, classes)
        logits = self.input_ops["logits"].forward()
        y_onehot = self.input_ops["y"].forward()

        # (bs, classes) -> (bs, 1)
        self.logits_exp = np.exp(logits)
        self.logits_sum_exp = np.sum(self.logits_exp, axis=-1, keepdims=True)

        # (bs, classes) -> (bs, 1)
        logits_y = y_onehot * logits
        logits_y = logits_y.sum(axis=-1).reshape(self.bs, 1)

        self.grad_y = -(logits - np.log(self.logits_sum_exp))

        ce_loss = y_onehot * self.grad_y
        ce_loss = ce_loss.sum(axis=-1).reshape(self.bs, 1)

        # sum_y should be all ones, but the unit test uses random numbers and
        # it is easier to compute real gradient sthat to
        sum_y = np.sum(y_onehot, axis=1, keepdims=True)
        prob_y = self.logits_exp * (1/self.logits_sum_exp)
        grad_logits = -(y_onehot - prob_y * sum_y)

        self.grad_logits = grad_logits

        return ce_loss

    def _backward(self, input_op:OpBase):
        if id(input_op) == id(self.input_ops["y"]):
            # is anyone ever going to take gradients w.r.t. y values?
            return self.grad_y
        else:
            return self.grad_logits

    def __repr__(self):
        return f"CELoss {self.shape}"
