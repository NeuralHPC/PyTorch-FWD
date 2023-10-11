#!/bin/bash
#
#SBATCH -A <project>
#SBATCH --nodes=1
#SBATCH --job-name=diff_train
#SBATCH --output=./out/diff_train-%j.out  
#SBATCH --error=./out/diff_train-%j.err
#SBATCH --time=23:59:59
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --partition booster


module load CUDA
module load Python

source <path_to_environment>

export PYTHONPATH=.

# CIFAR10
# Approx one day to train, following facebook if increased batch_size by k, increase lr by factor of k**0.5. Default lr is 2e-4 for batch-size 128.
# Training
srun torchrun --standalone --nproc_per_node=4 scripts/train.py --batch-size 128 --seed 42 --time-steps 1000 --dataset="CIFAR10" --epochs 300
# Sampling
srun torchrun --standalone --nproc_per_node=4 scripts/sample.py --ckpt-path="<give path here>" --input-shape 32 --dataset="CIFAR10" --sampler="DDPM" --batch-size 2500
