
import numpy as np
import pickle

from models.param import LookupParameter
from utils.math_utils import *


class Optimizer(object):
    def update(self):
        if hasattr(self, 'l2_coeff'):
            self._l2_addhook()
        if hasattr(self, 'gc_norm'):
            self._gradclip_addhook()
        self._update()

    def _update(self, **kwargs):
        raise NotImplementedError

    def _prepare(self):
        pass

    def _l2_addhook(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if type(param) == LookupParameter:
                for i, idx in enumerate(param.grad_indices):
                    param.part_grads[i] += 2 * self.l2_coeff * param.data[idx]
            else:
                raise NotImplementedError

    def _gradclip_addhook(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if type(param) == LookupParameter:
                for i, idx in enumerate(param.grad_indices):
                    norm = np.linalg.norm(param.part_grads[i])
                    if norm > self.gc_norm:
                        param.part_grads[i] *= self.gc_norm / norm
            else:
                raise NotImplementedError

    def regist_params(self, params):
        self.params = params
        self._prepare()

    def set_l2_reg(self, coeff):
        self.l2_coeff = coeff

    def set_gradclip(self, norm):
        self.gc_norm = norm

    def save_opt(self, opt_path):
        with open(opt_path, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load_opt(cls, opt_path):
        with open(opt_path, 'rb') as f:
            opt = pickle.load(f)
        return opt


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def _update(self):
        for param in self.params.values():
            if type(param) == LookupParameter:
                idxs = param.grad_indices
                if len(idxs) != 0:
                    param.data[idxs] -= self.lr * np.array(param.part_grads)
            else:
                param.data -= self.lr * param.grad

class Adagrad(Optimizer):
    def __init__(self, lr, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.grad_history = {}

    def _update(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if type(param) == LookupParameter:
                idxs = param.grad_indices
                if len(idxs) != 0:
                    self.grad_history[p_name][idxs] += np.power(param.part_grads, 2)
                    param.data[idxs] -= self.lr * np.array(param.part_grads) / (np.sqrt(self.grad_history[p_name][idxs]) + self.eps)
            else:
                self.grad_history[p_name] += np.power(param.grad, 2)
                param.data -= self.lr * param.grad / (np.sqrt(self.grad_history[p_name]) + self.eps)

    def _init_grad_history(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            self.grad_history[p_name] = np.zeros_like(param.data)

    def _prepare(self):
        self._init_grad_history()
