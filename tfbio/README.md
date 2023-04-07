[![pipeline status](https://gitlab.com/cheminfIBB/tfbio/badges/master/pipeline.svg)](https://gitlab.com/cheminfIBB/tfbio/commits/master)
[![coverage report](https://gitlab.com/cheminfIBB/tfbio/badges/master/coverage.svg)](https://gitlab.com/cheminfIBB/tfbio/commits/master)
[![anaconda](https://anaconda.org/cheminfIBB/tfbio/badges/installer/conda.svg)](https://anaconda.org/cheminfIBB/tfbio/)

This repository contains `tfbio` - package with helper functions I frequently use when building neural networks for biological data.

At the moment `tfbio.data` contains functions to deal with 3D structures of proteins and small molecules.
`tfbio.net` is less domain-specific and contains NN building blocks implemented in Tensorflow, as well as custom summary operations I use to monitor network parameters during training.
