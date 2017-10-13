
from models.base_model import BaseModel
from models.param import LookupParameter
from utils.math_utils import *


class HolE(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs.pop('n_entity')
        self.n_relation = kwargs.pop('n_relation')
        self.dim = kwargs.pop('dim')
        self.margin = kwargs.pop('margin')
        mode = kwargs.pop('mode', 'pairwise')
        if mode == 'pairwise':
            self.compute_gradients = self._pairwisegrads
        elif mode == 'single':
            self.compute_gradients = self._singlegrads
        else:
            raise NotImplementedError

        self.params = {'e': LookupParameter(name='e', shape=(self.n_entity, self.dim)),
                       'r': LookupParameter(name='r', shape=(self.n_relation, self.dim))}

    def _pairwisegrads(self, pos_samples, neg_samples):
        raise NotImplementedError

    def _singlegrads(self, samples, ys):
        self.prepare()
        scores = self.cal_triplet_scores(samples)
        loss = softplus(-ys*scores)

        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        s_embs = self.pick_ent(subs)
        r_embs = self.pick_rel(rels)
        o_embs = self.pick_ent(objs)

        # compute gradients
        df = np.expand_dims(-ys * (1 - sigmoid(ys*scores)), axis=1)
        s_grads = circular_correlation(r_embs, o_embs) * df
        r_grads = circular_correlation(s_embs, o_embs) * df
        o_grads = circular_convolution(s_embs, r_embs) * df

        # TODO: unify how to passing the gradients
        ents = np.r_[subs, objs]
        self.params['e'].add_all_grads(ents, np.r_[s_grads, o_grads])
        self.params['r'].add_all_grads(rels, r_grads)

        self.params['e'].finalize()
        self.params['r'].finalize()

        return loss.mean()

    def _composite(self, sub_emb, rel_emb):
        return circular_convolution(sub_emb, rel_emb)

    def _cal_similarity(self, query, obj_emb):
        return np.sum(query * obj_emb, axis=1)

    def cal_scores(self, subs, rels):
        sub_emb = self.pick_ent(subs)
        rel_emb = self.pick_rel(rels)
        qs = self._composite(sub_emb, rel_emb)
        score_mat = qs.dot(self.params['e'].data.T)
        return score_mat

    def cal_scores_inv(self, rels, objs):
        obj_emb = self.pick_ent(objs)
        rel_emb = self.pick_rel(rels)
        qs_inv = circular_correlation(rel_emb, obj_emb)
        score_mat = qs_inv.dot(self.params['e'].data.T)
        return score_mat

    def cal_triplet_scores(self, samples):
        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        sub_emb = self.pick_ent(subs)
        rel_emb = self.pick_rel(rels)
        obj_emb = self.pick_ent(objs)
        qs = self._composite(sub_emb, rel_emb)
        return self._cal_similarity(qs, obj_emb)

    def pick_ent(self, ents):
        return self.params['e'].data[ents]

    def pick_rel(self, rels):
        return self.params['r'].data[rels]
