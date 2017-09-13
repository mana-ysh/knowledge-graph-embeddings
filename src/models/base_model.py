
import dill


class BaseModel(object):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def cal_rank(self, **kwargs):
        raise NotImplementedError

    # For max-margin loss
    def _pairwisegrads(self, **kwargs):
        raise NotImplementedError

    # For log-likelihood
    def _singlegrads(self, **kwargs):
        raise NotImplementedError

    def _composite(self, **kwargs):
        raise NotImplementedError

    def _cal_similarity(self, **kwargs):
        raise NotImplementedError

    def pick_ent(self, **kwargs):
        raise NotImplementedError

    def pick_rel(self, **kwargs):
        raise NotImplementedError

    def cal_scores(self, **kwargs):
        raise NotImplementedError

    def cal_scores_inv(self, **kwargs):
        raise NotImplementedError

    def cal_triplet_scores(self, **kwargs):
        raise NotImplementedError

    def zerograds(self):
        for param in self.params.values():
            param.clear()

    def prepare(self):
        self.zerograds()

    def save_model(self, model_path):
        with open(model_path, 'wb') as fw:
            dill.dump(self, fw)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            model = dill.load(f)
        return model
