.. ShaRP documentation master file, created by
   sphinx-quickstart on Wed Nov 29 14:42:49 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ShaRP's documentation!
=================================

**ShaRP (Shapley for Rankings and Preferences)** is a game theoretic approach
to explain the contributions of features to different aspects of a ranked
outcome. ShaRP builds on the Quantitative Input Influence framework, and can
compute the contributions of features for multiple Quantities of Interest,
including score, rank, pair-wise preference, and top-k (see reference [1]_ for
details).

Installation
------------


The easiest way to install ``sharp`` is using ``pip``::

    # Install latest release
    pip install -U xai-sharp

    # Install additional dependencies (matplotlib) for plotting
    pip install -U "xai-sharp[optional]"

    # Install from source (may be unstable)
    pip install -U git+https://github.com/DataResponsibly/ShaRP

Alternatively, you can also clone the repository and install it from source::
    
    # Clone and switch to the project's directory
    git clone https://github.com/DataResponsibly/ShaRP.git
    cd ShaRP

    # Basic install
    pip install .

    # Installs project requirements and the research package. Dependecy group
    # "all" will also install the dependency groups shown below.
    pip install ".[optional,tests,docs]"

This generally only recommended if you intend to contribute to sharp, or extend it
somehow.

Reference
---------

.. [1] Pliatsika, V., Fonseca, J., Wang, T., Stoyanovich, J. "ShaRP:
   Explaining Rankings with Shapley Values." Under submission.


.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   auto_examples/index
   api
   genindex
