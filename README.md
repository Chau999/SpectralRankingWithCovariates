# Spectral Ranking with Covariates

This repository contains the code for the project **Spectral Ranking with Covariates**.



### Algorithms and where to find them

You will find implementations of the following ranking algorithms from the following scripts:

```
src/spektrankers.py
# Our proposed methods
├── SVDRankerNormal (SVDRank)
├── SVDRankerCov (SVDCovRank)
├── SVDRankerKCov (Kernelised SVDCovRank)
├── SerialRank (SerialRank)
├── CSerialRank (C-SerialRank)
├── CCARank (CCRank)
├── KCCARank (Kernelised CCRank)
# Spectral ranking benchmarks
├── RankCentrality (Rank Centrality) 
└── DiffusionRankCentrality (Regularised Rank Centrality)

# Probabilistic ranking benchmarks
src/prefkrr.py
└── PreferentialKRR (Bradley Terry with GP link)
src/baselines.py
├── BradleyTerryRanker (Bradley Terry Model)
└── Pairwise_LogisticRegression (Bradley Terry with logistic regressoin)

```

The algorithms used in this repo came primarily out of the following papers. If you use them in your research we would appreciate a citation to our paper:
```
@inproceedings{chau2022spectral,
  title={Spectral ranking with covariates},
  author={Chau, Siu Lun and Cucuringu, Mihai and Sejdinovic, Dino},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={70--86},
  year={2022},
  organization={Springer}
}
```

