# PLS-PF-S

This repository contains the supplementary numerical material and MATLAB
implementation associated with the paper

**Subspace methods for min-max problems**

by Morteza Kimiaei, Shima Shabani, and Michael Breuß.

The repository contains the MATLAB implementation of the proposed
line-search projection-based subspace methods (`PLS-S`) and their fixed-step
counterparts (`PF-S`), the learning-based monotone-equation test environment,
the comparison methods used in the numerical experiments, and the
supplementary material.

The repository is organized into four main parts.

## 1. Supplementary material

The folder `suppMat/` contains

    suppMat/suppMat.pdf

The supplementary material provides additional background and structural
analysis, including the asymptotic and canonical OGDA-type direction
structures, the complete construction and validation of the learning-based
test problems, and detailed numerical comparisons among the PLS-S, PF-S,
and OGDA-type methods.

## 2. Monotone-equation test environment

The folder `TE/` contains

    TE/TE.mat
    TE/genTE.m

The file `TE.mat` stores the MATLAB structure `TE`, which contains the
100 learning-based nonlinear monotone-equation test problems used in the
numerical experiments.

The test problems are generated from three application classes:

- Multi-Agent Reinforcement Learning (MARL),
- Robust Adversarial Learning (RAL),
- Generative Adversarial Networks (GANs).

All problems are unconstrained and have the form

    E(x) = 0,    x in R^n.

The script

    TE/genTE.m

generates the complete test environment using the fixed random seed 42 and
performs the validation checks described in the supplementary material.

To load the test environment in MATLAB, use

```matlab
load('TE/TE.mat','TE')
