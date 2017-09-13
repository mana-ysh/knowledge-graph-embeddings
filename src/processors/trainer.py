
import copy
import os
import time

from utils.dataset import *


class Trainer(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop('model')
        self.opt = kwargs.pop('opt')
        self.n_epoch = kwargs.pop('epoch')
        self.batchsize = kwargs.pop('batchsize')
        self.logger = kwargs.pop('logger')
        self.log_dir = kwargs.pop('model_dir')
        self.evaluator = kwargs.pop('evaluator')
        self.valid_dat = kwargs.pop('valid_dat')
        self.save_step = kwargs.pop('save_step')
        self.cur_epoch = 0
        self.model_path = os.path.join(self.log_dir, self.model.__class__.__name__)

    def _setup(self):
        self.logger.info('setup trainer...')
        self.opt.regist_params(self.model.params)
        self.best_model = None

    def _finalize(self):
        assert self.n_epoch == self.cur_epoch
        if self.valid_dat:
            self.best_model.save_model(self.model_path + '.best')
            best_epoch, best_val = self.evaluator.get_best_info()
            self.logger.info('===== Best metric: {} ({} epoch) ====='.format(best_val, best_epoch))
        else:
            self.model.save_model(self.model_path + '.epoch{}'.format(self.n_epoch))

    def _validation(self):
        valid_start = time.time()
        res = self.evaluator.run(self.model, self.valid_dat)
        self.logger.info('evaluation metric in {} epoch: {}'.format(self.cur_epoch, res))
        self.logger.info('evaluation time in {} epoch: {}'.format(self.cur_epoch, time.time() - valid_start))

        cur_best_epoch, cur_best_val = self.evaluator.get_best_info()
        self.logger.info('< Current Best metric: {} ({} epoch) >'.format(cur_best_val, cur_best_epoch))
        if cur_best_epoch == self.cur_epoch:
            self.best_model = copy.deepcopy(self.model)
        return cur_best_epoch

    def fit(self, **kwargs):
        raise NotImplementedError


class PairwiseTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PairwiseTrainer, self).__init__(**kwargs)
        self.n_negative = kwargs.pop('n_negative')
        self.neg_generator = UniformNegativeGenerator(self.model.n_entity, self.n_negative)

    def fit(self, train_dat):
        assert type(train_dat) == TripletDataset
        self._setup()
        for epoch in range(self.n_epoch):
            start = time.time()
            self.cur_epoch += 1
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            for pos_triplets in train_dat.batch_iter(self.batchsize):
                neg_triplets = self.neg_generator.generate(pos_triplets)
                loss = self.model.compute_gradients(np.tile(pos_triplets, (self.n_negative, 1)), neg_triplets)
                self.opt.update()
                sum_loss += loss

            if self.valid_dat:  # run validation
                cur_best_epoch = self._validation()

            if (epoch+1) % self.save_step == 0:
                if self.valid_dat:
                    self.best_model.save_model(self.model_path+'.epoch{}'.format(cur_best_epoch))
                else:
                    self.model.save_model(self.model_path + '.epoch{}'.format(epoch+1))

            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

        self._finalize()


class SingleTrainer(Trainer):
    def __init__(self, **kwargs):
        super(SingleTrainer, self).__init__(**kwargs)
        self.n_negative = kwargs.pop('n_negative')
        self.neg_generator = UniformNegativeGenerator(self.model.n_entity, self.n_negative)

    def fit(self, train_dat):
        self._setup()
        if type(train_dat) == TripletDataset:
            self._fit_negative_sample(train_dat)
        else:
            raise NotImplementedError

    def _fit_negative_sample(self, train_dat):
        assert type(train_dat) == TripletDataset
        for epoch in range(self.n_epoch):
            start = time.time()
            self.cur_epoch += 1
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            for pos_triplets in train_dat.batch_iter(self.batchsize):
                neg_triplets = self.neg_generator.generate(pos_triplets)
                ys = np.concatenate((np.ones(len(pos_triplets)), -np.ones(len(neg_triplets))))
                loss = self.model.compute_gradients(np.r_[pos_triplets, neg_triplets], ys)
                self.opt.update()
                sum_loss += loss

            if self.valid_dat:  # run validation
                cur_best_epoch = self._validation()

            if (epoch+1) % self.save_step == 0:
                if self.valid_dat:
                    self.best_model.save_model(self.model_path+'.epoch{}'.format(cur_best_epoch))
                else:
                    self.model.save_model(self.model_path + '.epoch{}'.format(epoch+1))

            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

        self._finalize()


class NegativeGenerator(object):
    def __init__(self, n_ent, n_negative, train_graph=None):
        self.n_ent = n_ent
        self.n_negative = n_negative
        if train_graph:
            raise NotImplementedError
        self.graph = train_graph  # for preventing from including positive triplets as negative ones

    def generate(self, pos_triplets):
        """
        :return: neg_triplets, whose size is (length of positives \times n_sample , 3)
        """
        raise NotImplementedError


class UniformNegativeGenerator(NegativeGenerator):
    def __init__(self, n_ent, n_negative, train_graph=None):
        super(UniformNegativeGenerator, self).__init__(n_ent, n_negative, train_graph)

    def generate(self, pos_triplets):
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * self.n_negative
        neg_ents = np.random.randint(0, self.n_ent, size=sample_size)
        neg_triplets = np.tile(pos_triplets, (self.n_negative, 1))
        head_or_tail = 2 * np.random.randint(0, 2, sample_size)
        neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        return neg_triplets


