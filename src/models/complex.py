
import copy

from models.base_model import BaseModel
from models.param import LookupParameter
from utils.math_utils import *


class ComplEx(BaseModel):
    def __init__(self, **kwargs):
        self.model_config = copy.deepcopy(kwargs)
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

        self.params = {'e_re': LookupParameter(name='e_re', shape=(self.n_entity, self.dim)),
                       'e_im': LookupParameter(name='e_im', shape=(self.n_entity, self.dim)),
                       'r_re': LookupParameter(name='r_re', shape=(self.n_relation, self.dim)),
                       'r_im': LookupParameter(name='r_im', shape=(self.n_relation, self.dim))}

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

            p_s_re_embs, p_s_im_embs = self.pick_ent(pos_subs)
            p_r_re_embs, p_r_im_embs = self.pick_rel(pos_rels)
            p_o_re_embs, p_o_im_embs = self.pick_ent(pos_objs)
            n_s_re_embs, n_s_im_embs = self.pick_ent(neg_subs)
            n_r_re_embs, n_r_im_embs = self.pick_rel(neg_rels)
            n_o_re_embs, n_o_im_embs = self.pick_ent(neg_objs)

            _batchsize = len(pos_subs)

            p_s_re_grads = - (p_r_re_embs * p_o_re_embs + p_r_im_embs * p_o_im_embs)
            p_s_im_grads = - (p_r_re_embs * p_o_im_embs - p_r_im_embs * p_o_re_embs)
            p_r_re_grads = - (p_s_re_embs * p_o_re_embs + p_s_im_embs * p_o_im_embs)
            p_r_im_grads = - (p_s_re_embs * p_o_im_embs - p_s_im_embs * p_o_re_embs)
            p_o_re_grads = - (p_s_re_embs * p_r_re_embs - p_s_im_embs * p_r_im_embs)
            p_o_im_grads = - (p_s_re_embs * p_r_im_embs + p_s_im_embs * p_r_re_embs)

            n_s_re_grads = n_r_re_embs * n_o_re_embs + n_r_im_embs * n_o_im_embs
            n_s_im_grads = n_r_re_embs * n_o_im_embs - n_r_im_embs * n_o_re_embs
            n_r_re_grads = n_s_re_embs * n_o_re_embs + n_s_im_embs * n_o_im_embs
            n_r_im_grads = n_s_re_embs * n_o_im_embs - n_s_im_embs * n_o_re_embs
            n_o_re_grads = n_s_re_embs * n_r_re_embs - n_s_im_embs * n_r_im_embs
            n_o_im_grads = n_s_re_embs * n_r_im_embs + n_s_im_embs * n_r_re_embs

            # TODO: unify how to passing the gradients
            for idx in range(_batchsize):
                self.params['e_re'].add_grad(pos_subs[idx], p_s_re_grads[idx])
                self.params['e_im'].add_grad(pos_subs[idx], p_s_im_grads[idx])
                self.params['r_re'].add_grad(pos_rels[idx], p_r_re_grads[idx])
                self.params['r_im'].add_grad(pos_rels[idx], p_r_im_grads[idx])
                self.params['e_re'].add_grad(pos_objs[idx], p_o_re_grads[idx])
                self.params['e_im'].add_grad(pos_objs[idx], p_o_im_grads[idx])

                self.params['e_re'].add_grad(neg_subs[idx], n_s_re_grads[idx])
                self.params['e_im'].add_grad(neg_subs[idx], n_s_im_grads[idx])
                self.params['r_re'].add_grad(neg_rels[idx], n_r_re_grads[idx])
                self.params['r_im'].add_grad(neg_rels[idx], n_r_im_grads[idx])
                self.params['e_re'].add_grad(neg_objs[idx], n_o_re_grads[idx])
                self.params['e_im'].add_grad(neg_objs[idx], n_o_im_grads[idx])

        else:
            pass

        self.params['e_re'].finalize()
        self.params['e_im'].finalize()
        self.params['r_re'].finalize()
        self.params['r_im'].finalize()

        return loss.mean()

    def _singlegrads(self, samples, ys):
        """
        each element in ys must be \{-1, 1 \}
        """
        self.prepare()
        scores = self.cal_triplet_scores(samples)
        loss = softplus(-ys*scores)

        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        s_re_embs, s_im_embs = self.pick_ent(subs)
        r_re_embs, r_im_embs = self.pick_rel(rels)
        o_re_embs, o_im_embs = self.pick_ent(objs)

        # compute gradients
        df = np.expand_dims(-ys * (1 - sigmoid(ys*scores)), axis=1)
        s_re_grads = (r_re_embs * o_re_embs + r_im_embs * o_im_embs) * df
        s_im_grads = (r_re_embs * o_im_embs - r_im_embs * o_re_embs) * df
        r_re_grads = (s_re_embs * o_re_embs + s_im_embs * o_im_embs) * df
        r_im_grads = (s_re_embs * o_im_embs - s_im_embs * o_re_embs) * df
        o_re_grads = (s_re_embs * r_re_embs - s_im_embs * r_im_embs) * df
        o_im_grads = (s_re_embs * r_im_embs + s_im_embs * r_re_embs) * df

        # TODO: unify how to passing the gradients
        ents = np.r_[subs, objs]
        self.params['e_re'].add_all_grads(ents, np.r_[s_re_grads, o_re_grads])
        self.params['e_im'].add_all_grads(ents, np.r_[s_im_grads, o_im_grads])
        self.params['r_re'].add_all_grads(rels, r_re_grads)
        self.params['r_im'].add_all_grads(rels, r_im_grads)

        self.params['e_re'].finalize()
        self.params['e_im'].finalize()
        self.params['r_re'].finalize()
        self.params['r_im'].finalize()

        return loss.mean()

    def _composite(self, sub_re_emb, sub_im_emb, rel_re_emb, rel_im_emb, prefix=''):
        re_qs = sub_re_emb * rel_re_emb - sub_im_emb * rel_im_emb
        im_qs = sub_re_emb * rel_im_emb + sub_im_emb * rel_re_emb
        return re_qs, im_qs

    def _cal_similarity(self, re_query, im_query, obj_re_emb, obj_im_emb):
        return np.sum(re_query * obj_re_emb, axis=1) + np.sum(im_query * obj_im_emb, axis=1)

    def cal_scores(self, subs, rels):
        sub_re_emb, sub_im_emb = self.pick_ent(subs)
        rel_re_emb, rel_im_emb = self.pick_rel(rels)
        re_qs, im_qs = self._composite(sub_re_emb, sub_im_emb, rel_re_emb, rel_im_emb)
        score_mat = re_qs.dot(self.params['e_re'].data.T) + im_qs.dot(self.params['e_im'].data.T)
        return score_mat

    def cal_scores_inv(self, rels, objs):
        obj_re_emb, obj_im_emb = self.pick_ent(objs)
        rel_re_emb, rel_im_emb = self.pick_rel(rels)
        re_qs_inv = obj_re_emb * rel_re_emb + obj_im_emb * rel_im_emb
        im_qs_inv = obj_im_emb * rel_re_emb - obj_re_emb * rel_im_emb
        score_mat = re_qs_inv.dot(self.params['e_re'].data.T) + im_qs_inv.dot(self.params['e_im'].data.T)
        return score_mat

    def cal_triplet_scores(self, samples):
        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        sub_re_emb, sub_im_emb = self.pick_ent(subs)
        rel_re_emb, rel_im_emb = self.pick_rel(rels)
        obj_re_emb, obj_im_emb = self.pick_ent(objs)
        re_qs, im_qs = self._composite(sub_re_emb, sub_im_emb, rel_re_emb, rel_im_emb)
        return self._cal_similarity(re_qs, im_qs, obj_re_emb, obj_im_emb)

    def pick_ent(self, ents):
        return self.params['e_re'].data[ents], self.params['e_im'].data[ents]

    def pick_rel(self, rels):
        return self.params['r_re'].data[rels], self.params['r_im'].data[rels]
