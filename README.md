# manifold_vi
Implementation of a variational inference approximation to the Bayesian nonparametric model in https://arxiv.org/abs/2205.15717 for learning the density of data supported about a manifold.

## Prerequisites

This package requires the following python packages:

```python
numpy
torch
pyro
matplotlib
tqdm
```

Installation is not required - all modules are imported locally here.

## Usage

Look in the notebook `tutorial.ipynb` for usage instructions. See `CircularAutoregressiveRationalQuadraticSpline` in https://vincentstimper.github.io/normalizing-flows/references/#normflows.flows for more details about the normalising flow hyperparameters.

## Acknowledgements

This project was supervised by Paul Rosa and Judith Rousseau during my MSc in Statistical Science (2023). The code for implementing circular spline normalising flows was adapted from https://github.com/VincentStimper/normalizing-flows.
