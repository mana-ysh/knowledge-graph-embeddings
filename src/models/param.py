
from collections import defaultdict
import numpy as np

from utils.math_utils import random_xavier


# TODO: Abstract class
class Parameter(object):
    def __init__(self, name, shape, init_method):
        self.name = name
        self.shape = shape
        if init_method == 'xavier':
            self.data = random_xavier(self.shape)
        else:
            raise NotImplementedError


class LookupParameter(Parameter):
    def __init__(self, name, shape, init_method='xavier'):
        super(LookupParameter, self).__init__(name, shape, init_method)
        self.grad_indices = None
        self.part_grads = None
        self.dim = shape[1]
        if len(self.shape) == 2:
            self.idx2grad = defaultdict(lambda: np.zeros(self.dim))
        elif len(self.shape) == 3:  # for RESCAL
            assert self.shape[1] == self.shape[2]
            self.idx2grad = defaultdict(lambda: np.zeros((self.dim, self.dim)))
        else:
            raise

    def add_grad(self, idx, grad):
        self.idx2grad[idx] += grad

    def add_all_grads(self, indices, grads):
        # TODO: Fix
        [self.add_grad(i, g) for i, g in zip(indices, grads)]

    def clear(self):
        if len(self.shape) == 2:
            self.idx2grad = defaultdict(lambda: np.zeros(self.dim))
        else:
            self.idx2grad = defaultdict(lambda: np.zeros((self.dim, self.dim)))

    def finalize(self):
        self.grad_indices = list(self.idx2grad.keys())
        self.part_grads = list(self.idx2grad.values())
        if not self.grad_indices:
            self.grad_indices = []
            self.part_grads = []
