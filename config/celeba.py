"""Config for CelebA and CelebHQ datasets."""

from typing import Dict, Any

celeba_path: str = '<path_to_celeba_dataset>'
celebahq_path: str = '<path_to_celebahq_dataset>'



class CELEBA64:
    def __init__(self) -> None:
        self.dataset_config: Dict[str, Any] = {
            'random_flip': True,
            'num_workers': 48, # Feel free to change this
            'dataset': 'CELEBA',
            'resize': 64,
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
            'num_workers': 48, # Feel free to change this
            'dataset': 'CELEBAHQ',
            'resize': 64,
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
            'num_workers': 48, # Feel free to change this
            'dataset': 'CELEBAHQ',
            'resize': 128,
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