
import argparse
from datetime import datetime
import logging
import numpy as np
import os

from processors.trainer import PairwiseTrainer, SingleTrainer
from processors.evaluator import Evaluator
from processors.optimizer import SGD, Adagrad
from utils.dataset import TripletDataset, Vocab


np.random.seed(46)
DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H:%M')))


def train(args):
    # setting for logging
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.log, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # TODO: develop the recording of arguments in logging
    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{:>10} -----> {}'.format(arg, val))

    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)
    n_entity, n_relation = len(ent_vocab), len(rel_vocab)

    # preparing data
    logger.info('preparing data...')
    train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab)
    valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None

    if args.filtered:
        logger.info('loading whole graph...')
        from utils.graph import TensorTypeGraph
        whole_graph = TensorTypeGraph.load_from_raw(args.graphall, ent_vocab, rel_vocab)
    else:
        whole_graph = None

    if args.opt == 'sgd':
        opt = SGD(args.lr)
    elif args.opt == 'adagrad':
        opt = Adagrad(args.lr)
    else:
        raise NotImplementedError

    if args.l2_reg > 0:
        opt.set_l2_reg(args.l2_reg)
    if args.gradclip > 0:
        opt.set_gradclip(args.gradclip)

    logger.info('building model...')
    if args.method == 'complex':
        from models.complex import ComplEx
        model = ComplEx(n_entity=n_entity,
                        n_relation=n_relation,
                        margin=args.margin,
                        dim=args.dim,
                        mode=args.mode)
    elif args.method == 'distmult':
        from models.distmult import DistMult
        model = DistMult(n_entity=n_entity,
                         n_relation=n_relation,
                         margin=args.margin,
                         dim=args.dim,
                         mode=args.mode)
    elif args.method == 'transe':
        from models.transe import TransE
        model = TransE(n_entity=n_entity,
                       n_relation=n_relation,
                       margin=args.margin,
                       dim=args.dim,
                       mode=args.mode)
    elif args.method == 'hole':
        from models.hole import HolE
        model = HolE(n_entity=n_entity,
                     n_relation=n_relation,
                     margin=args.margin,
                     dim=args.dim,
                     mode=args.mode)
    elif args.method == 'rescal':
        from models.rescal import RESCAL
        model = RESCAL(n_entity=n_entity,
                       n_relation=n_relation,
                       margin=args.margin,
                       dim=args.dim,
                       mode=args.mode)
    elif args.method == 'analogy':
        from models.analogy import ANALOGY
        model = ANALOGY(n_entity=n_entity,
                        n_relation=n_relation,
                        margin=args.margin,
                        dim=args.dim,
                        cp_ratio=args.cp_ratio,
                        mode=args.mode)
    else:
        raise NotImplementedError

    evaluator = Evaluator(args.metric, args.nbest, args.filtered, whole_graph) if args.valid or args.synthetic else None
    if args.filtered and args.valid:
        evaluator.prepare_valid(valid_dat)
    if args.mode == 'pairwise':
        trainer = PairwiseTrainer(model=model, opt=opt, save_step=args.save_step,
                                  batchsize=args.batch, logger=logger,
                                  evaluator=evaluator, valid_dat=valid_dat,
                                  n_negative=args.negative, epoch=args.epoch,
                                  model_dir=args.log)
    elif args.mode == 'single':
        trainer = SingleTrainer(model=model, opt=opt, save_step=args.save_step,
                                batchsize=args.batch, logger=logger,
                                evaluator=evaluator, valid_dat=valid_dat,
                                n_negative=args.negative, epoch=args.epoch,
                                model_dir=args.log)
    else:
        raise NotImplementedError

    trainer.fit(train_dat)

    logger.info('done all')


if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')
    p.add_argument('--mode', default='single', type=str, help='training mode ["pairwise", "single"]')

    # dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--train', type=str, help='training data')
    p.add_argument('--valid', type=str, help='validation data')

    # model
    p.add_argument('--method', default='complex', type=str, help='method ["complex", "distmult", "transe", "hole", "rescal", "analogy"]')
    p.add_argument('--epoch', default=300, type=int, help='number of epochs')
    p.add_argument('--batch', default=128, type=int, help='batch size')
    p.add_argument('--lr', default=0.05, type=float, help='learning rate')
    p.add_argument('--dim', default=200, type=int, help='dimension of embeddings')
    p.add_argument('--margin', default=1., type=float, help='margin in max-margin loss for pairwise training')
    p.add_argument('--negative', default=10, type=int, help='number of negative samples for pairwise training')
    p.add_argument('--opt', default='adagrad', type=str, help='optimizer ["sgd", "adagrad"]')
    p.add_argument('--l2_reg', default=0.0001, type=float, help='L2 regularization')
    p.add_argument('--gradclip', default=5, type=float, help='gradient clipping')
    p.add_argument('--save_step', default=100, type=int, help='epoch step for saving model')

    # model specific arguments
    p.add_argument('--cp_ratio', default=0.5, type=float, help="ratio of complex's dimention in ANALOGY")

    # evaluation
    p.add_argument('--metric', default='mrr', type=str, help='evaluation metrics ["mrr", "hits"]')
    p.add_argument('--nbest', default=None, type=int, help='n-best for hits metric')
    p.add_argument('--filtered', action='store_true', help='use filtered metric')
    p.add_argument('--graphall', type=str, help='all graph file for filtered evaluation')

    # others
    p.add_argument('--log', default=DEFAULT_LOG_DIR, type=str, help='output log dir')

    args = p.parse_args()

    train(args)
