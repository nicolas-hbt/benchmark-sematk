# Sem@K: Is my knowledge graph embedding model semantic-aware?

## Abstract

## Datasets
This repository contains most of the datasets used in [Sem@K: Is my knowledge graph embedding model semantic-aware?](link) for the semantic awareness evaluation of knowledge graph embedding models.

For YAGO4-19K, please visit [![DOI](https://zenodo.org/badge/576727654.svg)](https://zenodo.org/badge/latestdoi/576727654).

## Description

The ``datasets/`` folder contains the following datasets: ``FB15K237-ET/``, ``DB93K/``, ``YAGO3-37K``, ``YAGO4-19K``, ``Codex-S``, ``Codex-M``, and ``WN18RR``.

``FB15K237-ET/``, ``DB93K/``, ``YAGO3-37K``, and ``YAGO4-19K`` are **schema-defined**.

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
* ``rel2heads.pkl``: pickle file associating a relation ID to all of the entity IDs that have been observed as head in the training, validation and test set
* ``rel2tails.pkl``: pickle file associating a relation ID to all of the entity IDs that have been observed as tail in the training, validation and test set


Additionally, the schema-defined ``FB15K237-ET/``, ``DB93K/``, ``YAGO3-37K``, and ``YAGO4-19K`` have the following files:


* ``ent2class.pkl``: pickle file associating an entity ID to the list of the IDs of its instantiated classes. Note that instantiated classes are computed taking into account the transitive closure of the instantiation and subsumption relations.
* ``class2ents.pkl``: pickle file associating a class ID to the list of the IDs of its instances. Note that instances are computed taking into account the transitive closure of the instantiation and subsumption relations.
* ``subclassof2id.pkl``: pickle file associating a class ID to the list of the IDs of its direct superclasses
* ``rel2dom.pkl``: pickle file associating a relation ID to its domain ID
* ``rel2range.pkl``: pickle file associating a relation ID to its range ID
