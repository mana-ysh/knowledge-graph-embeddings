
import math
import numpy as np


def random_xavier(size):
    assert len(size) < 4
    # for RESCAL
    if len(size) == 3:
        assert size[1] == size[2]
    dim = size[1]
    bound = math.sqrt(6) / math.sqrt(2*dim)
    return np.random.uniform(-bound, bound, size=size)


def max_margin(pos_scores, neg_scores):
    return np.maximum(0, 1 - (pos_scores - neg_scores))


def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5


def softplus(x):
    return np.maximum(0, x)+np.log(1+np.exp(-np.abs(-x)))


def circular_convolution(v1, v2):
    freq_v1 = np.fft.fft(v1)
    freq_v2 = np.fft.fft(v2)
    return np.fft.ifft(np.multiply(freq_v1, freq_v2)).real


def circular_correlation(v1, v2):
    freq_v1 = np.fft.fft(v1)
    freq_v2 = np.fft.fft(v2)
    return np.fft.ifft(np.multiply(freq_v1.conj(), freq_v2)).real
