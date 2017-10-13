
from models.base_model import BaseModel
from models.param import LookupParameter
from utils.math_utils import *


class RESCAL(BaseModel):
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
                       'r_mat': LookupParameter(name='r_mat', shape=(self.n_relation, self.dim, self.dim))}

    def _pairwisegrads(self, pos_samples, neg_samples):
        assert pos_samples.shape == neg_samples.shape
        self.prepare()
        p_scores = self.cal_triplet_scores(pos_samples)
        n_scores = self.cal_triplet_scores(neg_samples)

        loss = max_margin(p_scores, n_scores)
        idxs = np.where(loss > 0)[0]
        if len(idxs) != 0:
            # TODO: inefficient calculation
            pos_subs, pos_rels, pos_objs = pos_samples[idxs, 0], pos_samples[idxs, 1], pos_samples[idxs, 2]
            neg_subs, neg_rels, neg_objs = neg_samples[idxs, 0], neg_samples[idxs, 1], neg_samples[idxs, 2]

            p_s_embs = self.pick_ent(pos_subs)
            p_r_mats = self.pick_rel(pos_rels)
            p_o_embs = self.pick_ent(pos_objs)
            n_s_embs = self.pick_ent(neg_subs)
            n_r_mats = self.pick_rel(neg_rels)
            n_o_embs = self.pick_ent(neg_objs)

            _batchsize = len(pos_subs)

            p_s_grads = - np.matmul(p_r_mats, np.expand_dims(p_o_embs, axis=2)).reshape(_batchsize, self.dim)
            p_r_grads = - np.matmul(np.expand_dims(p_s_embs, axis=2), np.expand_dims(p_o_embs, axis=1))
            p_o_grads = - self._composite(p_s_embs, p_r_mats)
            n_s_grads = np.matmul(n_r_mats, np.expand_dims(n_o_embs, axis=2)).reshape(_batchsize, self.dim)
            n_r_grads = np.matmul(np.expand_dims(n_s_embs, axis=2), np.expand_dims(n_o_embs, axis=1))
            n_o_grads = self._composite(n_s_embs, n_r_mats)

            for idx in range(_batchsize):
                self.params['e'].add_grad(pos_subs[idx], p_s_grads[idx])
                self.params['r_mat'].add_grad(pos_rels[idx], p_r_grads[idx])
                self.params['e'].add_grad(pos_objs[idx], p_o_grads[idx])
                self.params['e'].add_grad(neg_subs[idx], n_s_grads[idx])
                self.params['r_mat'].add_grad(neg_rels[idx], n_r_grads[idx])
                self.params['e'].add_grad(neg_objs[idx], n_o_grads[idx])

        else:
            pass

        self.params['e'].finalize()
        self.params['r_mat'].finalize()

        return loss.mean()

    def _singlegrads(self, samples, ys):
        raise NotImplementedError('Only pairwise setting is available')

    def _composite(self, sub_emb, rel_mat):
        _batchsize = len(sub_emb)
        return np.matmul(np.expand_dims(sub_emb, axis=1), rel_mat).reshape(_batchsize, self.dim)

    def _cal_similarity(self, query, obj_emb):
        return np.sum(query * obj_emb, axis=1)

    def cal_scores(self, subs, rels):
        sub_emb = self.pick_ent(subs)
        rel_mat = self.pick_rel(rels)
        qs = self._composite(sub_emb, rel_mat)
        score_mat = qs.dot(self.params['e'].data.T)
        return score_mat

    def cal_scores_inv(self, rels, objs):
        _batchsize = len(rels)
        obj_emb = self.pick_ent(objs)
        rel_mat = self.pick_rel(rels)
        qs_inv = np.matmul(rel_mat, np.expand_dims(obj_emb, axis=2)).reshape(_batchsize, self.dim)
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
        return self.params['r_mat'].data[rels]
