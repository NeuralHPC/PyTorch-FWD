Diffusion on CelebA dataset.

# Requirements:

Along with the requirements.txt please install jax corresponding to local cuda version.

# Dataset:
Steps:
1. Download the CelebA-HQ dataset (e.g., from here: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download)
2. Please provide paths the extracted dataset path to data-dir argument

# Training

To run Diffusion on MNIST
```
PYTHONPATH=. python scripts/train_diffuse_mnist.py --batch-size 100 --seed 42 --epochs 400 --data-dir <path to data>
```
Diffusion on CelebAHQ
```
PYTHONPATH=. python scripts/train_diffuse_celebA.py --batch-size 10 --seed 42 --epochs 200 --data-dir <path to data>
```
