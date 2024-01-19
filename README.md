# ShaRP

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

**NOTE: DO NOT USE THIS METHOD. IT IS NOT IMPLEMENTED YET**

The easiest way to install ml-research is using ``pip`` :

    pip install -U sharp

Or ``conda`` :

    conda install -c conda-forge sharp

The documentation includes more detailed [installation
instructions](https://sharp.readthedocs.io/en/latest/getting-started.html).

### Installing from source

The following commands should allow you to setup the development version of the
project with minimal effort:

    # Clone the project.
    git clone https://github.com/joaopfonseca/sharp.git
    cd sharp

    # Create and activate an environment 
    make environment 
    conda activate sharp # Assuming you are have conda set up

    # Install project requirements and the research package. Dependecy group
    # "all" will also install the dependency groups shown below.
    pip install .[optional,tests,docs] 
