Fréchet Wavelet Distance: A Domain-Agnostic Metric for Image Generation
=======================================================================

`Lokesh Veeramacheneni <https://lokiv.dev>`__\ *\ 1,*\ `Moritz
Wolter <https://www.wolter.tech/>`__\ *\ 1,*\ `Hilde
Kuehne <https://hildekuehne.github.io/>`__\ *\ 1,2, and*\ `Juergen
Gall <https://pages.iai.uni-bonn.de/gall_juergen/>`__\ *\ 1*

1 University of Bonn, 2 MIT-IBM Watson AI Lab.

[`Archive <https://arxiv.org/pdf/2312.15289>`__] [`Project
Page <https://lokiv.dev/frechet_wavelet_distance/>`__]

|Workflow| |License| |CodeStyle|

**Keywords:** Frechet Distance, Wavelet Packet Transform, Frechet
Inception Distance, Diffusion, GAN, ImageNet, Image generation metrics.

**Abstract:** Modern metrics for generative learning like Fréchet
Inception Distance (FID) demonstrate impressive performance. However,
they suffer from various shortcomings, like a bias towards specific
generators and datasets. To address this problem, we propose the Fréchet
Wavelet Distance (FWD) as a domain-agnostic metric based on Wavelet
Packet Transform (:math:`\mathcal{W}_p`). FWD provides a sight across a
broad spectrum of frequencies in images with a high resolution, along
with preserving both spatial and textural aspects. Specifically, we use
:math:`\mathcal{W}_p` to project generated and dataset images to packet
coefficient space. Further, we compute Fréchet distance with the
resultant coefficients to evaluate the quality of a generator. This
metric is general-purpose and dataset-domain agnostic, as it does not
rely on any pre-trained network while being more interpretable because
of frequency band transparency. We conclude with an extensive evaluation
of a wide variety of generators across various datasets that the
proposed FWD is able to generalize and improve robustness to domain
shift and various corruptions compared to other metrics.

.. container::

   <img src=“./images/fwd_computation.png”, width=“100%”/>

:hammer_and_wrench: Installation
================================

Clone the repository using

.. code:: bash

   git clone git@github.com:Uni-Bonn-Attention-Research/frechet_wavelet_distance.git
   cd ./frechet_wavelet_distance
   pip install .

:test_tube: Requirements
========================

All the requirements are specified in
`requirements.txt <https://github.com/Uni-Bonn-Attention-Research/diffusion/blob/pytorch/requirements.txt>`__
file.

:school_satchel: Usage
======================

.. code:: bash

    python -m fwd <path to dataset> <path to generated images>

Here are the other arguments and defaults used.

.. code:: bash

   python -m fwd --help

usage: fwd.py [-h] [–batch-size BATCH_SIZE] [–num-processes
NUM_PROCESSES] [–save-packets] [–wavelet WAVELET] [–max_level MAX_LEVEL]
[–log_scale] path path

positional arguments: path Path to the generated images or path to .npz
statistics file.

options: -h, –help show this help message and exit –batch-size
BATCH_SIZE Batch size for wavelet packet transform. (default: 128)
–num-processes NUM_PROCESSES Number of multiprocess. (default: None)
–save-packets Save the packets as npz file. (default: False) –wavelet
WAVELET Choice of wavelet. (default: sym5) –max_level MAX_LEVEL wavelet
decomposition level (default: 4) –log_scale Use log scaling for
wavelets. (default: False)

::

   We conduct all the experiments with `Haar` wavelet with transformation/decomposition level of `4` for `256x256` image.

   In future, we plan to release the jax-version of this code.

   # :paperclip: Citation
   If you use this repository in your research, please cite using the following bibtex entry.

Replace bibtex here after paper is updated. \``\`

:star: Acknowledgments
======================

The code is built with inspiration from
`Pytorch-FID <https://github.com/mseitzer/pytorch-fid>`__. We use
`PyTorch Wavelet
Toolbox <https://github.com/v0lta/PyTorch-Wavelet-Toolbox>`__ for
Wavelet Packet Transform implementation. We recommend to have a look at
these repositories.

:construction: ToDO
===================

-  ☒ Project Page
-  ☐ WPKL Code setup
-  ☐ PIP package
-  ☐ JAX version

:heavy_plus_sign: Wavelet Power KL-Divergence (WPKL)
====================================================

We also experimented with KLDivergence version and found that
KLDivergence suffers from scaling issues. Please refer to `link
here <https://google.com/>`__ here for its usage.

.. |Workflow| image:: https://github.com/Uni-Bonn-Attention-Research/frechet_wavelet_distance/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/Uni-Bonn-Attention-Research/frechet_wavelet_distance/actions/workflows/tests.yml
.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |CodeStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
