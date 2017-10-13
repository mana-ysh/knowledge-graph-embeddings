
import argparse

from processors.evaluator import Evaluator
from utils.dataset import TripletDataset, Vocab
from utils.graph import *


def test(args):
    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)

    # preparing data
    test_dat = TripletDataset.load(args.data, ent_vocab, rel_vocab)

    print('loading model...')
    if args.method == 'complex':
        from models.complex import ComplEx as Model
    elif args.method == 'distmult':
        from models.distmult import DistMult as Model
    elif args.method == 'transe':
        from models.transe import TransE as Model
    elif args.method == 'hole':
        from models.hole import HolE as Model
    else:
        raise NotImplementedError

    if args.filtered:
        print('loading whole graph...')
        from utils.graph import TensorTypeGraph
        whole_graph = TensorTypeGraph.load_from_raw(args.graphall, ent_vocab, rel_vocab)
    else:
        whole_graph = None
    evaluator = Evaluator('all', None, args.filtered, whole_graph)
    if args.filtered:
        evaluator.prepare_valid(test_dat)
    model = Model.load_model(args.model)

    all_res = evaluator.run_all_matric(model, test_dat)
    for metric in sorted(all_res.keys()):
        print('{:20s}: {}'.format(metric, all_res[metric]))


if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')

    # dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--data', type=str, help='test data')
    p.add_argument('--filtered', action='store_true', help='use filtered metric')
    p.add_argument('--graphall', type=str, help='all graph file for filtered evaluation')

    # model
    p.add_argument('--method', default=None, type=str, help='method ["complex", "distmult", "transe", "hole"]')
    p.add_argument('--model', type=str, help='trained model path')

    args = p.parse_args()
    test(args)
