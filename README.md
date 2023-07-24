Diffusion on CelebA dataset.

# Requirements:

Along with the requirements.txt please install jax corresponding to local cuda version.

# Dataset:
Steps:
1. CelebA dataset can be downloaded [official drive link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) or [website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
2. Download and extract img_align_celeba.zip from Img folder.
3. Then download list_eval_partition.txt from Eval and finally download labels file (identity_CelebA.txt) from Anno folder.
4. Please provide paths to corresponding variables in src/util.py functions get_batched_celebA_paths and batch_loader.
