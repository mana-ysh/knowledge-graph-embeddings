
import numpy as np
from scipy.sparse import lil_matrix

from utils.dataset import *


class TensorTypeGraph(object):
    def __init__(self, triple_dat, n_ent, n_rel):
        self.rel2mat = [lil_matrix((n_ent, n_ent)) for _ in range(n_rel)]
        for triple in triple_dat.batch_iter(1, rand_flg=False):
            sub, rel, obj = triple[0]
            self.rel2mat[rel][sub, obj] = 1.0

    def search_obj_id(self, sub, rel):
        return np.where(self.rel2mat[rel][sub].todense() == 1.0)[1]

    def search_sub_id(self, rel, obj):
        return np.where(self.rel2mat[rel][:, obj].todense() == 1.0)[0]

    @classmethod
    def load_from_raw(cls, data_path, ent_v, rel_v):
        triples = TripletDataset.load(data_path, ent_v, rel_v)
        return TensorTypeGraph(triples, len(ent_v), len(rel_v))
