# Sem@K: Is my knowledge graph embedding model semantic-aware?

## Abstract
Using knowledge graph embedding models (KGEMs) is a popular approach for predicting links in knowledge graphs (KGs). Traditionally, the performance of KGEMs for link prediction is assessed using rank-based metrics, which evaluate their ability to give high scores to ground-truth entities. However, the literature claims that the KGEM evaluation procedure would benefit from adding supplementary dimensions to assess.
That is why, in this paper, we extend our previously introduced metric Sem@K that measures the capability of models to predict valid entities w.r.t. domain and range constrains.
In particular, we consider a broad range of KGs and take their respective characteristics into account to propose different versions of Sem@K.
We also perform an extensive study of KGEM semantic awareness.
Our experiments show that Sem@K provides a new perspective on KGEM quality. Its joint analysis with rank-based metrics offer different conclusions on the predictive power of models. Regarding Sem@K, some KGEMs are inherently better than others, but this semantic superiority is not indicative of their performance w.r.t. rank-based metrics. In this work, we generalize conclusions about the relative performance of KGEMs w.r.t. rank-based and semantic-oriented metrics at the level of families of models. The joint analysis of the aforementioned metrics gives more insight into the peculiarities of each model. This work paves the way for a more comprehensive evaluation of KGEM adequacy for specific downstream tasks.

## Forewords
This repository contains code and datasets used in [Sem@K: Is my knowledge graph embedding model semantic-aware?](link) for the semantic awareness evaluation of knowledge graph embedding models. In particular, we re-implemented TransE, TransH, DistMult, ComplEx, SimplE, ConvE, and ConvKB ourselves. For R-GCN and CompGCN, please refer to [R-GCN-LinkPrediction](https://github.com/toooooodo/RGCN-LinkPrediction), and [CompGCN](https://github.com/malllabiisc/CompGCN), respectively. We thank the authors for providing their implementations.

## Datasets

The ``datasets/`` folder contains the following datasets: ``FB15K237/``, ``DB93K/``, ``YAGO3-37K``, ``YAGO4-18K``, ``Codex-S``, ``Codex-M``, and ``WN18RR``.

``FB15K237/``, ``DB93K/``, ``YAGO3-37K``, and ``YAGO4-18K`` are **schema-defined**.

``Codex-S``, ``Codex-M``, and ``WN18RR`` are **schemaless**.


For all the datasets contained in the ``datasets/`` folder, we provide the following files:


* ``train2id.txt``: train set in which each line is tab-separated and of the form ``subjectID relationID objectID``
* ``valid2id.txt``: valid set in which each line is tab-separated and of the form ``subjectID relationID objectID``
* ``test2id.txt``: test set in which each line is tab-separated and of the form ``subjectID relationID objectID``

* ``ent2id.pkl``: pickle file associating an entity URI to its equivalent ID
* ``rel2id.pkl``: pickle file associating a relation URI to its equivalent ID
* ``class2id.pkl``: pickle file associating a class URI to its equivalent ID
* ``head2triples.pkl``: pickle file associating an entity ID to all of the ``relationID objectID`` pairs for which this entity ID has been observed as a subject in the training, validation and test set
* ``tail2triples.pkl``: pickle file associating an entity ID to all of the ``relationID subjectID`` pairs for which this entity ID has been observed as an object in the training, validation and test set
* ``head2triples_inv.pkl``: pickle file associating an entity ID to all of the ``relationID objectID`` pairs for which this entity ID has been observed as a subject in the training, validation and test set. This file contains inverse relations and is used for ConvE
* ``tail2triples_inv.pkl``: pickle file associating an entity ID to all of the ``relationID subjectID`` pairs for which this entity ID has been observed as an object in the training, validation and test set. This file contains inverse relations and is used for ConvE


Additionally, the schema-defined ``FB15K237/``, ``DB93K/``, ``YAGO3-37K``, and ``YAGO4-18K`` have the following files:


* ``ent2class.pkl``: pickle file associating an entity ID to the list of the IDs of its instantiated classes. Note that instantiated classes are computed taking into account the transitive closure of the instantiation and subsumption relations.
* ``class2ents.pkl``: pickle file associating a class ID to the list of the IDs of its instances. Note that instances are computed taking into account the transitive closure of the instantiation and subsumption relations.
* ``subclassof2id.pkl``: pickle file associating a class ID to the list of the IDs of its direct superclasses
* ``rel2dom.pkl``: pickle file associating a relation ID to its domain ID
* ``rel2range.pkl``: pickle file associating a relation ID to its range ID

## Full Pipeline

The full range of experiments can be run at once with:


```bash
sh full.sh
```


## Single Experiment

Alternatively, we provide the files ``dataset.py``, ``trainer.py``, ``tester.py``, ``models.py``, and ``main.py``.
A single experiment can be run with specific command-line parameters fed into the main program. For example:


```bash
python main.py -pipeline train -dataset YAGO4-18K -model TransE -ne 400 -save_each 10 -lr 0.001 -metrics all -reg 0.00001 -dim 100 -setting CWA -sem both
```


## Plots

We provide the code for reproducing the plots found in the paper. Once your models are trained (e.g. after running the full pipeline as explained above), you can copy-paste the results of your models into ``plots/models/``. Alternatively, we provide the output results as csv file.
All the plots can be run at once with the following command:


```bash
python plots/plots.py
```

