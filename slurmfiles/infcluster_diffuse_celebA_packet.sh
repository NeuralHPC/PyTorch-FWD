#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=ca_diff_w
#SBATCH --output=out/diff_train-%j.out
#SBATCH --error=out/diff_train-%j.err
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate jax
which python
pip install -r requirements.txt

PYTHONPATH=. python scripts/train_diffuse_celebA.py --batch-size 100 --seed 22 --epochs 400 --data-dir ./data/celebA --wavelet-loss