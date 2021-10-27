# Strategies for fitting logistic regression for positive and unlabeled data revisited

This repository contains code used to obtain results presented in the paper.

Repository structure:
- `requirements.txt` contains dependencies necessary to run the code (install using `pip install -r requirements.txt`),
- `simple_test.py` contains examples of usage for implemented methods,
- `data/` directory contains benchmark datasets used for tests,
- `datasets.py` and `data_preprocessing.py` contain code used to load and preprocess datasets,
- `optimization/` directory constains code of used methods:
    - CCCP, DCCP, MM, joint, naive and weighted classifiers, as well as oracle method as the reference method,
    - `c_estimation` subdirectory contains implementations of label frequency estimation methods: EN, TIcE and a helper estimator based directly on class prior value, without fitting,
    - `functions/` subdirectory contains risk functions, wrappers used to obtain statistics and some utitlity functions,
    - `metrics/` subdirectory contains scripts used to measure metrics.
- `results/` subdirectory contains scripts used to perform tests and draw various plots. In particular, `tests_wrt_c.py` calculates metrics used in the paper and `process_detailed_results.py` plots them.

MOSEK optimizer is used in DCCP method - in order to use it, a valid license must be present. Visit [https://www.mosek.com/](https://www.mosek.com/) for details.