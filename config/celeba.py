"""Config for CelebA and CelebHQ datasets."""

from typing import Dict, Any

celeba_path: str = None
celebahq_path: str = None


class CELEBA64:
    def __init__(self) -> None:
        self.dataset_config: Dict[str, Any] = {
            'random_flip': True,
            'num_workers': 48,  # Feel free to change this
            'dataset': 'CELEBA',
            'resize': 64,
            'mean': [x / 255.0 for x in [129.058, 108.485, 97.622]],
            'std': [x / 255.0 for x in [78.338, 73.131, 72.970]],
        }

        self.model_config: Dict[str, Any] = {
            'in_c': 3,
            'out_c': 3,
            'model_c': 128,
            'num_res_blocks': 2,
            'attn_res': tuple([16]),
            'dropout': 0.0,
            'channel_mult': tuple([1, 2, 2, 2, 4]),
            'num_classes': 1000,
            'num_heads': 4,
            'num_heads_ups': 4,
            'use_scale_shift_norm': True,
            'input_size': 64
        }

        self.optimizer_config: Dict[str, Any] = {
            'lr': 2e-4,
            'clip_grad_norm': 1.0  # TODO: Check this again.
        }

        self.data_dir: str = celeba_path


class CELEBAHQ64:
    def __init__(self) -> None:
        self.dataset_config: Dict[str, Any] = {
            'random_flip': True,
            'num_workers': 48,  # Feel free to change this
            'dataset': 'CELEBAHQ',
            'resize': 64,
            'mean': [x / 255.0 for x in [131.810, 106.258, 92.634]],
            'std': [x / 255.0 for x in [76.332, 69.183, 67.954]],
        }

        self.model_config: Dict[str, Any] = {
            'in_c': 3,
            'out_c': 3,
            'model_c': 128,
            'num_res_blocks': 2,
            'attn_res': tuple([16, 8]),
            'dropout': 0.0,
            'channel_mult': tuple([1, 2, 2, 4]),
            'num_classes': 1000,
            'num_heads': 4,
            'num_heads_ups': 4,
            'use_scale_shift_norm': True,
            'input_size': 64
        }

        self.optimizer_config: Dict[str, Any] = {
            'lr': 1e-4,
            'clip_grad_norm': 1.0  # TODO: Check this again.
        }

        self.data_dir: str = celebahq_path


class CELEBAHQ128:
    def __init__(self) -> None:
        self.dataset_config: Dict[str, Any] = {
            'random_flip': True,
            'num_workers': 48,  # Feel free to change this
            'dataset': 'CELEBAHQ',
            'resize': None,
            'mean': [x / 255.0 for x in [131.810, 106.258, 92.634]],
            'std': [x / 255.0 for x in [76.332, 69.183, 67.954]],
        }

        self.model_config: Dict[str, Any] = {
            'in_c': 3,
            'out_c': 3,
            'model_c': 128,
            'num_res_blocks': 3,
            'attn_res': tuple([16, 8]),
            'dropout': 0.0,
            'channel_mult': tuple([1, 2, 2, 4, 4]),
            'num_classes': 1000,
            'num_heads': 4,
            'num_heads_ups': 4,
            'use_scale_shift_norm': True,
            'input_size': 128
        }

        self.optimizer_config: Dict[str, Any] = {
            'lr': 1e-4,
            'clip_grad_norm': 1.0  # TODO: Check this again.
        }

        self.data_dir: str = celebahq_path
