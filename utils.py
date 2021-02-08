import random
import os

import numpy as np
import torch
import torch.nn as nn
import yaml


def load_config(config_path):
    """Load config file from `config_path`.

    Args:
        config_path (str): Configuration file path, which must be in `config` dir, e.g.,
            `./config/inner_dir/example.yaml` and `config/inner_dir/example`.
    
    Returns:
        config (dict): Configuration dict.
        inner_dir (str): Directory between `config/` and configuration file. If `config_path`
           doesn't contain `inner_dir`, return empty string.
        config_name (str): Configuration filename.
    """
    assert os.path.exists(config_path)
    config_hierarchy = config_path.split("/")
    if config_hierarchy[0] != ".":
        if config_hierarchy[0] != "config":
            raise RuntimeError(
                "Configuration file {} must be in config dir".format(config_path)
            )
        if len(config_hierarchy) > 2:
            inner_dir = os.path.join(*config_hierarchy[1:-1])
        else:
            inner_dir = ""
    else:
        if config_hierarchy[1] != "config":
            raise RuntimeError(
                "Configuration file {} must be in config dir".format(config_path)
            )
        if len(config_hierarchy) > 3:
            inner_dir = os.path.join(*config_hierarchy[2:-1])
        else:
            inner_dir = ""
    print("Load configuration file from {}:".format(config_path))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_name = config_hierarchy[-1].split(".yaml")[0]

    return config, inner_dir, config_name


def gen_poison_idx(dataset, target_label, poison_ratio=None):
    poison_idx = np.zeros(len(dataset))
    train = dataset.train
    for (i, t) in enumerate(dataset.targets):
        if train and poison_ratio is not None:
            if random.random() < poison_ratio and t == target_label:
                poison_idx[i] = 1
        else:
            if t != target_label:
                poison_idx[i] = 1

    return poison_idx


class NormalizeByChannelMeanStd(nn.Module):
    """Normalizing the input to the network.
    """

    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor.sub(mean).div(std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)
