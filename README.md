Diffusion on CelebA dataset.

# Requirements:

Along with the requirements.txt please install jax corresponding to local cuda version.

# Dataset:
$\bullet$ Download the CelebA-HQ dataset (e.g., from here: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download)
## Important:
For CIFAR10 the dataset is automatically downloaded by torchvision.
In case of Celeba and CelebaHQ, please fill the corresponding datapaths in `config/celeba.py`.


# Training and sampling (Uninode):
To train diffusion model on uninode
```
PYTHONPATH=. torchrun --standalone --nproc_per_node=<num_gpus> scripts/train.py --batch-size <bs> --epochs <n_epochs> --time-steps 1000 --dataset=<dataset_name>
```
For sampling on uninode
```
PYTHONPATH=. torchrun --standalone --nproc_per_node=<num_gpus> scripts/sample.py --ckpt-path <checkpoint directory> --input-shape <input_img_height> --dataset=<dataset_name> --sampler="DDPM" --batch-size <bs> --diff-steps=1000
```
The supported dataset options are "CIFAR10", "CELEBA64", "CELEBAHQ64", "CELEBAHQ128" and supported sampler options are "DDPM", "DDIM".



# FID
Computing the FID
```
PYTHONPATH=. python scripts/compute_fid.py --data-dir <dataset_path> --input-size <64> --sample-sir <sampled_activations>
```
