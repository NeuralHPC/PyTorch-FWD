"""Config for CIFAR10."""

from typing import Dict, Any, Union


class CIFAR10_Config:
    def __init__(self)-> None:
        self.dataset_config: Dict[str, Any] = {
            "random_flip": True,
            "num_workers": 48,  # Feel free to change this
            "dataset": "CIFAR10",
            "resize": None,
            "mean": [x / 255.0 for x in [125.3, 123.0, 113.9]],
            "std": [x / 255.0 for x in [63.0, 62.1, 66.7]],
        }

        self.model_config: Dict[str, Any] = {
            "in_c": 3,
            "out_c": 3,
            "model_c": 128,
            "num_res_blocks": 2,
            "attn_res": tuple([16]),
            "dropout": 0.1,
            "channel_mult": tuple([1, 2, 2, 2]),
            "num_classes": 1000,
            "num_heads": 4,
            "num_heads_ups": 4,
            "use_scale_shift_norm": True,
            "input_size": 32,
        }

        self.optimizer_config: Dict[str, Any] = {"lr": 2e-4, "grad_clip_norm": 1.0}

        self.data_dir: Union[str, None] = None
