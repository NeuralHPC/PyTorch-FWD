#!/bin/bash
#
#SBATCH -A <project>
#SBATCH --nodes=2
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

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Name of allocated nodes: $SLURM_JOB_NODELIST
echo Number of allocated nodes: $SLURM_JOB_NUM_NODES
echo Cluster name: $SLURM_CLUSTER_NAME
echo Node IP: $head_node_ip
echo Nodes: $nodes_array

export PYTHONPATH=.


# CIFAR10
# Approx one day to train, following facebook if increased batch_size by k, increase lr by factor of k**0.5. Default lr is 2e-4 for batch-size 128.
# Training normal
# echo "CIFAR10 training normal"
# srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node:29500 scripts/train.py --batch-size 2048 --seed 42 --time-steps 1000 --dataset="CIFAR10" --epochs 300 --distribute --lr 8e-4
# Training Wavelet loss (Unnormalized)
echo "CIFAR10 training wavelet loss"
srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node:29500 scripts/train.py --batch-size 2048 --seed 42 --time-steps 1000 --dataset="CIFAR10" --epochs 300 --distribute --lr 8e-4 --loss-type 'PACKET' --max-level=1

# Sampling
# echo "CIFAR10 sampling"
# srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node:29500 scripts/sample.py --ckpt-path="<give path here>" --input-shape 32 --dataset="CIFAR10" --sampler="DDPM" --batch-size 2500 --distribute

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"