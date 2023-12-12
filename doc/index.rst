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

ShaRP can be installed from source::

    pip install git+https://github.com/DataResponsibly/ShaRP.git

Alternatively::
    
    git clone https://github.com/DataResponsibly/ShaRP.git
    cd ShaRP
    pip install .

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
