# OKRidge <!-- omit in toc -->

[![docs](https://readthedocs.org/projects/okridge/badge/?version=latest)](https://readthedocs.org/projects/okridge/?badge=latest)
[![pypi](https://img.shields.io/pypi/v/okridge?color=blue)](https://pypi.org/project/okridge/)
[![license](https://img.shields.io/badge/License-BSD-brightgreen)](https://github.com/jiachangliu/OKRidge/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/okridge)](https://pepy.tech/project/okridge)
[![downloads](https://img.shields.io/pypi/dm/okridge)](https://pypistats.org/packages/okridge)
[![arxiv badge](https://img.shields.io/badge/arXiv-2304.06686-red)](https://arxiv.org/abs/2304.06686)

This repository contains source code to our NeurIPS 2023 paper:

[**OKRidge: Scalable Optimal k-Sparse Ridge Regression**](https://arxiv.org/abs/2304.06686)

- Documentation: [https://okridge.readthedocs.io](https://okridge.readthedocs.io)
- GitHub: [https://github.com/jiachangliu/OKRidge](https://github.com/jiachangliu/OKRidge)
- PyPI: [https://pypi.org/project/okridge/](https://pypi.org/project/okridge/)
- Free and open source software: [BSD license](https://github.com/jiachangliu/OKRidge/blob/main/LICENSE)

# Table of Content <!-- omit in toc -->
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Introduction

We consider the following optimization problem:

$$\min_{\mathbf{\beta}} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \mathbf{\beta})^2 + \lambda_2 \lVert \mathbf{\beta} \rVert_2^2 \quad \text{s.t.} \quad \lVert \mathbf{\beta} \rVert_0 \leq k$$

Optimal k-sparse ridge regression is a crucial ML problem that has many applications in statistics, machine learning, and data mining.
However, the problem is NP-hard, and existing algorithms are either slow (using commercial MIP solvers) or suboptimal (using convex or nonconvex regularizers to approximate $\ell_0$).
We propose a novel algorithm, OKRidge, that can solve the problem to provable optimality in a scalable manner.

OKRidge is based on the [branch-and-bound](https://en.wikipedia.org/wiki/Branch_and_bound) framework.
The insight leading to the computational efficiency comes from a novel lower bound calculation involving, first, a saddle point formulation, and from there, either solving (i) a linear system or (ii) using an [ADMM](https://stanford.edu/~boyd/admm.html)-based approach, where the proximal operators can be efficiently evaluated by solving another linear system and an [isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression#:~:text=In%20statistics%20and%20numerical%20analysis,to%20the%20observations%20as%20possible.) problem.
We also propose a method to warm-start our solver, which leverages a beam search.

Experimentally, our methods attain provable optimality with run times that are orders of magnitude faster than those of the existing MIP formulations solved by the commercial solver Gurobi.


# Installation

```bash
$ pip install okridge
```

# Usage

Please see the [example.ipynb](https://github.com/jiachangliu/OKRidge/blob/main/docs/example.ipynb) jupyter notebook on GitHub for a detailed tutorial on how to use OKRidge in a python environment.

```python
k = 10 # cardinality constraint
lambda2 = 0.1 # l2 regularization parameter
gap_tol = 1e-4 # optimality gap tolerance
verbose = True # print out the progress
time_limit = 180 # time limit in seconds

BnB_optimizer = BNBTree(X=X, y=y, lambda2=lambda2)

upper_bound, betas, optimality_gap, max_lower_bound, running_time = BnB_optimizer.solve(k = k, gap_tol = gap_tol, verbose = verbose, time_limit = time_limit)
```

# Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

# License

`okridge` was created by Jiachang Liu. It is licensed under the terms of the BSD 3-Clause license.

# Credits <!-- omit in toc -->

`okridge` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

# Citing Our Work <!-- omit in toc -->

If you find our work useful in your research, please consider citing the following paper:

```BibTeX
@inproceedings{liu2023okridge,
  title={OKRidge: Scalable Optimal k-Sparse Ridge Regression},
  author={Liu, Jiachang and Rosen, Sam and Zhong, Chudi and Rudin, Cynthia},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```