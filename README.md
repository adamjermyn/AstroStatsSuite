# Overview

AstroStatsSuite is a collection of Python implementations of the statistical methods discussed in the accompanying [methods paper](https://arxiv.org/abs/1801.06545) along with scripts for benchmarking them.

The folder *Fit Code* contains the statistical method files.
Most of these are meant for performing 0-E (regression with no errors) or 1-E regression (e.g. regression with errors in a single variable).
These are provided each as a single Python script with a method that takes as input matching lists of `x`- and `y`-coordinates. 
When available, errors must be for the observations in `y` and are provided as an an additional parameter.
In some cases technique-specific additional parameters are also required.

Each of these methods returns the following four objects:
1. The predicted values of the function represented by the pairs `(x[i], y[i])` at each input `x[i]`. For some methods this is precisely `y[i]`, but methods are not guaranteed to reproduce their inputs exactly.
2. Auxiliary data specific to the regression technique.
3. A list of indices in x at which the regression will evaluate a prediction of `y`. For most methods this is simply `range(len(x))`, but some methods explicitly exclude evaluations at the edges of the dataset, in which case the corresponding indices are omittted.
4. A function which represents the result of the regression. This takes as input x-values and produces as output predicted y-values. Input may be either numpy arrays or scalars.

In addition the *Fit Code* folder contains a file *funcs.py* which holds implementations of the sample functions tested in the methods paper, as well as an implementation of our GIRAfFE algorithm for regression when there are errors present in both variables. This is available in the file *deconFit.py*.

The folder *Benchmark Code* contains scripts which were used to generate the benchmarks shown in the methods paper. These scripts write output into a folder *Benchmark Output* which must be created at the top level (e.g. the same level as the other folders). The results of this are then analyzed by the scripts in *Plot Scripts*.

# Referencing AstroStatsSuite

AstroStatsSuite is free to use, but if you use it for academic purposes please include a citation to the methods paper:

Nonparametric Methods in Astronomy: Think, Regress, Observe—Pick Any Three - Charles L. Steinhardt and Adam S. Jermyn -  arXiv:1801.06545

The BibTex entry for this is included below:

```

@ARTICLE{2018PASP..130b3001S,
   author = {{Steinhardt}, C.~L. and {Jermyn}, A.~S.},
    title = "{Nonparametric Methods in Astronomy: Think, Regress, Observe{\mdash}Pick Any Three}",
  journal = {\pasp},
archivePrefix = "arXiv",
   eprint = {1801.06545},
 primaryClass = "astro-ph.IM",
     year = 2018,
    month = feb,
   volume = 130,
   number = 2,
    pages = {023001},
      doi = {10.1088/1538-3873/aaa22a},
   adsurl = {http://adsabs.harvard.edu/abs/2018PASP..130b3001S},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```