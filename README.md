# knowledge-graph-embeddings

Python Implementations of Embedding-based methods for Knowledge Base Completion tasks, mainly inspired by [scikit-kge](https://github.com/mnick/scikit-kge) and [complex](https://github.com/ttrouill/complex).

## List of methods
- RESCAL [Nickel+. 2011]
- TransE [Bordes+. 2013]
- DistMult [Yang+. 2015]
- HolE [Nicklel+. 2016] 
  - This model is equivalent to ComplEx, and the computation cost of ComplEx is lower than of HolE.
- ComplEx [Trouillon+. 2016]
- ANALOGY [Liu+. 2017] (not implemented yet)
  - This model can be regarded as a hybrid between DistMult and RESCAL.


## Run to train and test

For training...

```
▶  python train.py -h
usage: Link prediction models [-h] [--mode MODE] [--ent ENT] [--rel REL]
                              [--train TRAIN] [--valid VALID]
                              [--method METHOD] [--epoch EPOCH]
                              [--batch BATCH] [--lr LR] [--dim DIM]
                              [--margin MARGIN] [--negative NEGATIVE]
                              [--opt OPT] [--l2_reg L2_REG]
                              [--gradclip GRADCLIP] [--save_step SAVE_STEP]
                              [--metric METRIC] [--nbest NBEST] [--filtered]
                              [--graphall GRAPHALL] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           training mode ["pairwise", "single"]
  --ent ENT             entity list
  --rel REL             relation list
  --train TRAIN         training data
  --valid VALID         validation data
  --method METHOD       method ["complex", "distmult", "transe", "hole",
                        "rescal"]
  --epoch EPOCH         number of epochs
  --batch BATCH         batch size
  --lr LR               learning rate
  --dim DIM             dimension of embeddings
  --margin MARGIN       margin in max-margin loss for pairwise training
  --negative NEGATIVE   number of negative samples for pairwise training
  --opt OPT             optimizer ["sgd", "adagrad"]
  --l2_reg L2_REG       L2 regularization
  --gradclip GRADCLIP   gradient clipping
  --save_step SAVE_STEP
                        epoch step for saving model
  --metric METRIC       evaluation metrics ["mrr", "hits"]
  --nbest NBEST         n-best for hits metric
  --filtered            use filtered metric
  --graphall GRAPHALL   all graph file for filtered evaluation
  --log LOG             output log dir
```


For testing...

```
▶  python test.py -h
usage: Link prediction models [-h] [--ent ENT] [--rel REL] [--data DATA]
                              [--filtered] [--graphall GRAPHALL]
                              [--method METHOD] [--model MODEL]

optional arguments:
  -h, --help           show this help message and exit
  --ent ENT            entity list
  --rel REL            relation list
  --data DATA          test data
  --filtered           use filtered metric
  --graphall GRAPHALL  all graph file for filtered evaluation
  --method METHOD      method ["complex", "distmult", "transe", "hole",
                       "rescal"]
  --model MODEL        trained model path
```

## Experiments

### WordNet (WN18)

| Models | MRR (flt) | MRR (raw) | Hits@1 (flt) | Hits@3 (flt) | Hits@10 (flt) |
|:-----------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| ComplEx* | 94.1 | 58.7 | 93.6 | 94.5 | 94.7 |
| ComplEx | 94.3 | 58.2 | 94.0 | 94.6 | 94.8 |

### FreeBase (FB15k)
| Models | MRR (flt) | MRR (raw) | Hits@1 (flt) | Hits@3 (flt) | Hits@10 (flt) |
|:-----------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| ComplEx* | 69.2 | 24.2 | 59.9 | 75.9 | 84.0 |
| ComplEx | 69.5 | 24.2 | 59.8 | 76.9 | 85.0 |

\* means the results reported from the original papers 

## Dependencies
* numpy
* scipy


## References

* Bordes, A.; Usunier, N.; Garcia-Duran, A.; Weston, J.; and Yakhnenko, O. 2013. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems (NIPS). 

* Liu, H.; Wu, Y.; and Yang, Y. 2017. Analogical inference for multi-relational embeddings. In Proceedings of the 34th International Conference on Machine Learning (ICML).

* Nickel, M.; Rosasco, L.; and Poggio, T. 2016. Holographic embeddings of knowledge graphs. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI’16.

* Nickel, M.; Tresp, V.; and Kriegel, H.-P. 2011. A threeway model for collective learning on multi-relational data. In International Conference on Machine Learning (ICML-11), ICML ’11,

* Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, E.; and Bouchard, G. 2016. Complex embeddings for simple link prediction. In International Conference on Machine Learning (ICML).

* Yang, B.; Yih, W.; He, X.; Gao, J.; and Deng, L. 2015. Embedding entities and relations for learning and inference in knowledge bases. International Conference on Learning Representations 2015.
