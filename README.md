# Adagio for Things
This repository contains a Tensorflow implementation of the VYAE autoencoder for interaction sound effects. Details and application examples can be found in the reference below and on the [accompanying website](https://www.audiolabs-erlangen.de/resources/2022-AVAR-InteractionSounds). 

## Reference
Simon Schwär, Meinard Müller, and Sebastian J. Schlecht: **A Variational Y-Autoencoder for Disentangling Gesture and Material of Interaction Sounds.** In *AES 4th International Conference on Audio for Virtual and Augmented Reality (AES AVAR) – submitted for peer review*, Redmond, WA
, USA, 2022.

## Installation
Required software packages: tensorflow (tested with versions 2.6 and 2.7), numpy (tested with versions 1.19 and 1.21).
The demonstration notebook further requires: jupyterlab, matplotlib, [librosa](https://github.com/librosa/librosa)

## Usage
The repository contains the saved weights of the model trained with the Spoon/Bowl dataset. Please see the notebook `explore_trained_model.ipynb` for a usage example.

The file `train_model.py` can be used to train the model on other data. Some API documentation can be found in the `vyae` package.
