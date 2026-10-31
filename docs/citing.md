# Citing apxchol

If you use apxchol in published work, please cite the software and the
method papers it implements.

## Software

```bibtex
@software{apxchol,
  author  = {Kulkov, Oleksandr and Baumann, Yves and Kyng, Rasmus},
  title   = {{apxchol}: scalable approximate Cholesky preconditioning for Laplacian and SDDM systems},
  url     = {https://github.com/AlgOptGroup/apxchol},
  version = {X.Y.Z},
  license = {BSD-3-Clause},
}
```

Replace `X.Y.Z` with the version you used (`apxchol --version` or
`apxchol.__version__`). A Zenodo DOI for each release will be added here from
1.0 onwards. GitHub's "Cite this repository" button reads the same metadata from
[`CITATION.cff`](../CITATION.cff).

## Method

apxchol's approximate elimination follows Kyng and Sachdeva, its default
clique sampler (`gks`) is from Gao, Kyng and Spielman, and its parallel
elimination follows Baumann and Kyng.

```bibtex
@inproceedings{kyng2016approximate,
  author    = {Kyng, Rasmus and Sachdeva, Sushant},
  title     = {Approximate {G}aussian Elimination for {L}aplacians: Fast, Sparse, and Simple},
  booktitle = {2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS)},
  pages     = {573--582},
  year      = {2016},
  doi       = {10.1109/FOCS.2016.68},
}

@misc{gao2023robust,
  author        = {Gao, Yuan and Kyng, Rasmus and Spielman, Daniel A.},
  title         = {Robust and Practical Solution of {L}aplacian Equations by Approximate Elimination},
  year          = {2023},
  eprint        = {2303.00709},
  archivePrefix = {arXiv},
}

@inproceedings{baumann2024parallel,
  author    = {Baumann, Yves and Kyng, Rasmus},
  title     = {A Framework for Parallelizing Approximate {G}aussian Elimination},
  booktitle = {Proceedings of the 36th ACM Symposium on Parallelism in Algorithms and Architectures (SPAA)},
  pages     = {195--206},
  year      = {2024},
  doi       = {10.1145/3626183.3659987},
}
```
