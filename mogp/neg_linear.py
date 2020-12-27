import numpy as np
from GPy.mappings import Linear


class NegLinear(Linear):
    """
    A negative linear mapping

    .. math::

       F(\mathbf{x}) = -\mathbf{A} \mathbf{x})

    Arguments:
        input_dim (int): dimension of input
        output_dim (int): dimension of output
        name(str): name of mapping
    """

    def __init__(self, input_dim, output_dim, name='neg_linmap'):
        super(NegLinear, self).__init__(input_dim=input_dim, output_dim=output_dim, name=name)

    def f(self, X):
        return np.dot(X, -self.A)

    def update_gradients(self, dL_dF, X):
        self.A.gradient = np.dot(-X.T, dL_dF)

    def gradients_X(self, dL_dF):
        return np.dot(dL_dF, -self.A.T)
