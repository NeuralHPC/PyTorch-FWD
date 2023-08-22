Diffusion on CelebA dataset.

# Requirements:

Along with the requirements.txt please install jax corresponding to local cuda version.

# Dataset:
Steps:
1. Download the CelebA-HQ dataset (e.g., from here: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download)
2. Please provide paths the extracted dataset path to data-dir argument

# Training and sampling

To run Diffusion on MNIST
```
PYTHONPATH=. python scripts/train_diffuse_mnist.py --batch-size 100 --seed 42 --epochs 400 --data-dir <path to data>
```
Diffusion on CelebAHQ
```
PYTHONPATH=. python scripts/train_diffuse_celebA.py --batch-size 10 --seed 42 --epochs 200 --data-dir <path to data>
```
Sampling on CelebAHQ
```
PYTHONPATH=. python scripts/sample_diffuse_celebA.py --ckpt-path <checkpoint_path> --input-shape <64>
```
for input_shape argument please provide either height or width of the image. Currently only supports square images.<br>
if no seed was given, a random seed is selected. To provide seed use ```--args.seed <seed>```.<br>
Add ```--gif``` to the above command for visualizing the diffusion steps of one random label.
This script saves the mean and variance of InceptionV3 activations for the sampled images.

# FID
Computing the FID
```
PYTHONPATH=. python scripts/compute_fid.py --data-dir <dataset_path> --input-size <64> --sample-sir <sampled_activations>
```