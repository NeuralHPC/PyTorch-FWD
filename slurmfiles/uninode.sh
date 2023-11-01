#!/bin/bash
#
#SBATCH -A holistic-vid-westai
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

source /p/project/holistic-vid-westai/veeramacheneni2/venvs/mlpytorch/bin/activate

export PYTHONPATH=.

# srun torchrun --standalone --nproc_per_node=4 scripts/train.py --batch-size 128 --seed 42 --time-steps 1000 --epochs 100 --loss-type "PACKET" --lr 1e-5 --max-level 1 --wavelet "sym5" --dataset "CIFAR10"
srun torchrun --standalone --nproc_per_node=4 scripts/train.py --batch-size 128 --seed 42 --time-steps 1000 --epochs 100 --loss-type "PACKET" --lr 1e-5 --max-level 4 --wavelet "sym5" --dataset "CELEBAHQ256"
# srun torchrun --standalone --nproc_per_node=4 scripts/train.py --batch-size 128 --seed 42 --time-steps 1000 --epochs 100 --loss-type "PACKET" --lr 1e-5 --max-level 4 --wavelet "sym5" --dataset "CHURCH"
# srun torchrun --standalone --nproc_per_node=4 scripts/train.py --batch-size 128 --seed 42 --time-steps 1000 --epochs 100 --loss-type "PACKET" --lr 1e-5 --max-level 4 --wavelet "sym5" --dataset "BEDROOM"
