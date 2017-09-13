"""
TODO
- writing Item class for supporting fancy indexing in PathQueryDataset
"""

import numpy as np


# TODO: Abstract class
class Dataset(object):
    def __init__(self, samples):
        assert type(samples) == list or type(samples) == np.ndarray
        self._samples = samples if type(samples) == np.ndarray else np.array(samples)

    def __getitem__(self, item):
        return self._samples[item]

    def __len__(self):
        return len(self._samples)

    def batch_iter(self, batchsize, rand_flg=True):
        indices = np.random.permutation(len(self)) if rand_flg else np.arange(len(self))
        for start in range(0, len(self), batchsize):
            yield self[indices[start: start+batchsize]]

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        raise NotImplementedError


class TripletDataset(Dataset):
    def __init__(self, samples):
        super(TripletDataset, self).__init__(samples)

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj]))
        return TripletDataset(samples)


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    @classmethod
    def load(cls, vocab_path):
        v = Vocab()
        with open(vocab_path) as f:
            for word in f:
                v.add(word.strip())
        return v
