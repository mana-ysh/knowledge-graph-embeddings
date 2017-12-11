
from models.base_model import BaseModel
from models.param import LookupParameter
from utils.math_utils import *


class ANALOGY(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs.pop('n_entity')
        self.n_relation = kwargs.pop('n_relation')
        self.dim = kwargs.pop('dim')
        self.margin = kwargs.pop('margin')
        self.complex_ratio = kwargs.pop('cp_ratio')
        assert self.complex_ratio >= 0 and self.complex_ratio <= 1
        mode = kwargs.pop('mode', 'pairwise')
        if mode == 'pairwise':
            self.compute_gradients = self._pairwisegrads
        elif mode == 'single':
            self.compute_gradients = self._singlegrads
        else:
            raise NotImplementedError
        comp_dim = int(self.dim * self.complex_ratio)
        dist_dim = self.dim - comp_dim

        self.params = {'e_re': LookupParameter(name='e_re', shape=(self.n_entity, comp_dim)),
                       'e_im': LookupParameter(name='e_im', shape=(self.n_entity, comp_dim)),
                       'r_re': LookupParameter(name='r_re', shape=(self.n_relation, comp_dim)),
                       'r_im': LookupParameter(name='r_im', shape=(self.n_relation, comp_dim)),
                       'e': LookupParameter(name='e', shape=(self.n_entity, dist_dim)),
                       'r': LookupParameter(name='r', shape=(self.n_relation, dist_dim))}

    def _singlegrads(self, samples, ys):
        """
        each element in ys must be \{-1, 1 \}
        """
        self.prepare()
        scores = self.cal_triplet_scores(samples)
        loss = softplus(-ys*scores)

        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        s_re_embs, s_im_embs, s_embs = self.pick_ent(subs)
        r_re_embs, r_im_embs, r_embs = self.pick_rel(rels)
        o_re_embs, o_im_embs, o_embs = self.pick_ent(objs)

        # compute gradient
        df = np.expand_dims(-ys * (1 - sigmoid(ys*scores)), axis=1)
        s_re_grads = (r_re_embs * o_re_embs + r_im_embs * o_im_embs) * df
        s_im_grads = (r_re_embs * o_im_embs - r_im_embs * o_re_embs) * df
        r_re_grads = (s_re_embs * o_re_embs + s_im_embs * o_im_embs) * df
        r_im_grads = (s_re_embs * o_im_embs - s_im_embs * o_re_embs) * df
        o_re_grads = (s_re_embs * r_re_embs - s_im_embs * r_im_embs) * df
        o_im_grads = (s_re_embs * r_im_embs + s_im_embs * r_re_embs) * df
        s_grads = (r_embs * o_embs) * df
        r_grads = (s_embs * o_embs) * df
        o_grads = (s_embs * r_embs) * df

        ents = np.r_[subs, objs]
        self.params['e_re'].add_all_grads(ents, np.r_[s_re_grads, o_re_grads])
        self.params['e_im'].add_all_grads(ents, np.r_[s_im_grads, o_im_grads])
        self.params['r_re'].add_all_grads(rels, r_re_grads)
        self.params['r_im'].add_all_grads(rels, r_im_grads)
        self.params['e'].add_all_grads(ents, np.r_[s_grads, o_grads])
        self.params['r'].add_all_grads(rels, r_grads)

        self.params['e_re'].finalize()
        self.params['e_im'].finalize()
        self.params['r_re'].finalize()
        self.params['r_im'].finalize()
        self.params['e'].finalize()
        self.params['r'].finalize()

        return loss.mean()

    def _comp_composite(self, sub_re_emb, sub_im_emb, rel_re_emb, rel_im_emb):
        re_qs = sub_re_emb * rel_re_emb - sub_im_emb * rel_im_emb
        im_qs = sub_re_emb * rel_im_emb + sub_im_emb * rel_re_emb
        return re_qs, im_qs

    def _dist_composite(self, sub_emb, rel_emb):
        return np.multiply(sub_emb, rel_emb)

    # def _cal_similarity(self, re_query, im_query, obj_re_emb, obj_im_emb):
    #     return np.sum(re_query * obj_re_emb, axis=1) + np.sum(im_query * obj_im_emb, axis=1)

    def cal_scores(self, subs, rels):
        _batchsize = len(subs)
        sub_re_emb, sub_im_emb, sub_emb = self.pick_ent(subs)
        rel_re_emb, rel_im_emb, rel_emb = self.pick_rel(rels)
        re_qs, im_qs = self._comp_composite(sub_re_emb, sub_im_emb, rel_re_emb, rel_im_emb)
        comp_score_mat = re_qs.dot(self.params['e_re'].data.T) + im_qs.dot(self.params['e_im'].data.T)

        qs = self._dist_composite(sub_emb, rel_emb)
        dist_score_mat = qs.dot(self.params['e'].data.T)
        return comp_score_mat + dist_score_mat

    def cal_scores_inv(self, rels, objs):
        _batchsize = len(objs)
        obj_re_emb, obj_im_emb, obj_emb = self.pick_ent(objs)
        rel_re_emb, rel_im_emb, rel_emb = self.pick_rel(rels)
        re_qs_inv = obj_re_emb * rel_re_emb + obj_im_emb * rel_im_emb
        im_qs_inv = obj_im_emb * rel_re_emb - obj_re_emb * rel_im_emb
        comp_score_mat = re_qs_inv.dot(self.params['e_re'].data.T) + im_qs_inv.dot(self.params['e_im'].data.T)

        qs = self._dist_composite(obj_emb, rel_emb)
        dist_score_mat = qs.dot(self.params['e'].data.T)
        return comp_score_mat + dist_score_mat

    def cal_triplet_scores(self, samples):
        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        sub_re_emb, sub_im_emb, sub_emb = self.pick_ent(subs)
        rel_re_emb, rel_im_emb, rel_emb = self.pick_rel(rels)
        obj_re_emb, obj_im_emb, obj_emb = self.pick_ent(objs)

        # complex
        re_qs, im_qs = self._comp_composite(sub_re_emb, sub_im_emb, rel_re_emb, rel_im_emb)
        comp_score = np.sum(re_qs * obj_re_emb, axis=1) + np.sum(im_qs * obj_im_emb, axis=1)

        # distmult
        qs = self._dist_composite(sub_emb, rel_emb)
        dist_score = np.sum(qs * obj_emb, axis=1)

        return comp_score + dist_score

    def pick_ent(self, ents):
        return self.params['e_re'].data[ents], self.params['e_im'].data[ents], self.params['e'].data[ents]

    def pick_rel(self, rels):
        return self.params['r_re'].data[rels], self.params['r_im'].data[rels], self.params['r'].data[rels]
