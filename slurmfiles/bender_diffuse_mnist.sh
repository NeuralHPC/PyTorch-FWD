#!/bin/bash

#SBATCH --job-name=diff_train
#SBATCH --output=out/diff_train-%j.out
#SBATCH --error=out/diff_train-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH --partition=A100medium
#SBATCH --cpus-per-task=200
#SBATCH --mem 100000M

source ~/.bashrc
conda activate /home/mwolter1/miniconda3/envs/jax

PYTHONPATH=. python scripts/diffuse_mnist.py --batch-size 100 --seed 42 --epochs 800