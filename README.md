<div align="center">
    <h1 align="center">ShaRP</h1>
</div>

______________________________________________________________________

<p align="center">
<a href="https://github.com/DataResponsibly/ShaRP/actions/workflows/ci.yml"><img alt="Github Actions" src="https://github.com/DataResponsibly/ShaRP/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://codecov.io/gh/joaopfonseca/ml-research"><img alt="Codecov" src="https://codecov.io/gh/joaopfonseca/ml-research/branch/master/graph/badge.svg?token=J2EBA4YTMN"></a>
<a href="https://dataresponsibly.github.io/ShaRP/"><img alt="Documentation Status" src="https://github.com/DataResponsibly/ShaRP/actions/workflows/deploy-docs.yml/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"></a>

``ShaRP`` is an open source library with the implementation of the ShaRP
algorithm (Shapley for Rankings and Preferences), a framework that can be used
to explain the contributions of features to different aspects of a ranked
outcome, based on Shapley values.

## Installation

A Python distribution of version >= 3.9 is required to run this
project. ``ShaRP`` requires:

- numpy (>= 1.20.0)
- pandas (>= 1.3.5)
- scikit-learn (>= 1.2.0)
- ml-research (>= 0.4.2)

Some functions require Matplotlib (>= 2.2.3) for plotting.

### User Installation

The easiest way to install ``sharp`` is using ``pip`` :

    pip install -U xai-sharp

Or ``conda`` :

    conda install -c conda-forge xai-sharp

The documentation includes more detailed [installation
instructions](https://sharp.readthedocs.io/en/latest/getting-started.html).

### Installing from source

The following commands should allow you to setup the development version of the
project with minimal effort:

    # Clone the project.
    git clone https://github.com/DataResponsibly/sharp.git
    cd sharp

    # Create and activate an environment 
    make environment 
    conda activate sharp # Assuming you are have conda set up

    # Install project requirements and the research package. Dependecy group
    # "all" will also install the dependency groups shown below.
    pip install .[optional,tests,docs] 

## Citing ShaRP

If you use ``sharp`` in a scientific publication, we would appreciate citations to the following paper:

    @article{pliatsika2024sharp,
      title={ShaRP: Explaining Rankings with Shapley Values},
      author={Pliatsika, Venetia and Fonseca, Joao and Wang, Tilun and Stoyanovich, Julia},
      journal={arXiv preprint arXiv:2401.16744},
      year={2024}
    }
