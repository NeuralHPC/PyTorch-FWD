#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=ca_diff_w
#SBATCH --output=out/diff_train_cap-%j.out
#SBATCH --error=out/diff_train_cap-%j.err
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate jax
which python
pip install -r requirements.txt

PYTHONPATH=. python scripts/train_diffuse_celebA.py --batch-size 100 --seed 22 --epochs 600 --data-dir ./data/celebA --wavelet-loss