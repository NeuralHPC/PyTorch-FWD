# Fréchet Wavelet Distance: A Domain-Agnostic Metric for Image Generation

This repository contains code for Paper [update here](https://arxiv.org/html/2312.15289v1).


# Installation
Clone the repository using 
```
git clone git@github.com:Uni-Bonn-Attention-Research/diffusion.git
cd ./diffusion
```

# Requirements
All the requirements are specified in [requirements.txt](https://github.com/Uni-Bonn-Attention-Research/diffusion/blob/pytorch/requirements.txt) file.

# Usage
For simpler usage, we follow similar development pattern as [Pytorch-FID](https://github.com/mseitzer/pytorch-fid).
```
export PYTHONPATH=.
python src/fwd.py <path to dataset> <path to generated images>
```
Here are the other arguments and defaults used.
```
python src/fwd.py --help

usage: fwd.py [-h] [--batch-size BATCH_SIZE] [--num-processes NUM_PROCESSES] [--save-packets] [--wavelet WAVELET] [--max_level MAX_LEVEL] [--log_scale] path path

positional arguments:
  path                  Path to the generated images or path to .npz statistics file.

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for wavelet packet transform. (default: 128)
  --num-processes NUM_PROCESSES
                        Number of multiprocess. (default: None)
  --save-packets        Save the packets as npz file. (default: False)
  --wavelet WAVELET     Choice of wavelet. (default: sym5)
  --max_level MAX_LEVEL
                        wavelet decomposition level (default: 4)
  --log_scale           Use log scaling for wavelets. (default: False)
```
We conduct all the experiments with `Haar` wavelet with transformation/decomposition level of `4` for `256x256` image.

In future, we plan to release the jax-version of this code.

# Wavelet Power KL-Divergence (WPKL)
We also experimented with KLDivergence version and found that KLDivergence suffers from scaling issues.
Please refer to [link here](https://google.com/) here for its usage.

# Citation
If you use this repository in your research, please cite using the following bibtex entry.
```
Replace bibtex here after paper is updated.
```

# ToDO
- [ ] PIP package
- [ ] JAX version