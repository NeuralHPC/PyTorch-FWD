#!/bin/bash
#
#SBATCH -A holistic-vid-westai
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=diff_train
#SBATCH --output=out/diff_train-%j.out
#SBATCH --error=out/diff_train-%j.err
#SBATCH --time=10:00:00

module load Python
module load CUDA
ml CUDA/.12.0

export LD_LIBRARY_PATH=/p/software/juwelsbooster/stages/2023/software/CUDA/12.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/p/home/jusers/wolter1/juwels/project_drive/cudnn-linux-x86_64-8.9.1.23_cuda12-archive/lib:$LD_LIBRARY_PATH

source /p/home/jusers/wolter1/juwels/project_drive/jax_env/bin/activate

PYTHONPATH=. python scripts/diffuse_mnist.py --batch-size 100 --seed 42

# PYTHONPATH=. python scripts/diffuse_mnist.py --batch-size 50 --seed 43

# PYTHONPATH=. python scripts/diffuse_mnist.py --batch-size 50 --seed 44

# PYTHONPATH=. python scripts/diffuse_mnist.py --batch-size 50 --seed 45

