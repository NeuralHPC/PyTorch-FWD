#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH --job-name=diff_train2
#SBATCH --output=out/diff_train-%j.out
#SBATCH --error=out/diff_train-%j.err
#SBATCH --time=24:00:00

conda activate jax
which python

PYTHONPATH=. python scripts/train_diffuse_mnist.py --batch-size 50 --seed 22 --epochs 400