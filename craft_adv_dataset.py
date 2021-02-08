import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.cifar import CIFAR10
from model.network.resnet import resnet18

from utils import NormalizeByChannelMeanStd, load_config
from torchattacks import PGD

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/craft/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)

    args = parser.parse_args()
    config, _, config_name = load_config(args.config)

    train_transform = train_transform = transforms.Compose([transforms.ToTensor()])
    train_data = CIFAR10(config["dataset_dir"], transform=train_transform, train=True)
    train_loader = DataLoader(train_data, **config["loader"])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu = torch.cuda.current_device()
    print("Set GPU to: {}".format(args.gpu))
    model = resnet18()
    model = model.cuda(gpu)
    adv_ckpt = torch.load(config["adv_model_path"], map_location="cuda:{}".format(gpu))
    model.load_state_dict(adv_ckpt)
    print(
        "Load training state from the checkpoint {}:".format(config["adv_model_path"])
    )
    if config["normalization_layer"] is not None:
        normalization_layer = NormalizeByChannelMeanStd(**config["normalization_layer"])
        normalization_layer = normalization_layer.cuda(gpu)
        print("Add a normalization layer: {} before model".format(normalization_layer))
        model = nn.Sequential(normalization_layer, model)

    pgd_config = config["pgd"]
    print("Set PGD attacker: {}.".format(pgd_config))
    max_pixel = pgd_config.pop("max_pixel")
    for k, v in pgd_config.items():
        if k == "eps" or k == "alpha":
            pgd_config[k] = v / max_pixel
    attacker = PGD(model, **pgd_config)
    attacker.set_return_type("int")

    perturbed_img = torch.zeros((len(train_data), *config["size"]), dtype=torch.uint8)
    target = torch.zeros(len(train_data))
    i = 0
    for item in tqdm(train_loader):
        # Adversarially perturb image. Note that torchattacks will automatically
        # move `img` and `target` to the gpu where the attacker.model is located.
        img = attacker(item["img"], item["target"])
        perturbed_img[i : i + len(img), :, :, :] = img.permute(0, 2, 3, 1).detach()
        target[i : i + len(item["target"])] = item["target"]
        i += img.shape[0]

    if not os.path.exists(config["adv_dataset_dir"]):
        os.makedirs(config["adv_dataset_dir"])
    adv_data_path = os.path.join(
        config["adv_dataset_dir"], "{}.npz".format(config_name)
    )
    np.savez(adv_data_path, data=perturbed_img.numpy(), targets=target.numpy())
    print("Save the adversarially perturbed dataset to {}".format(adv_data_path))


if __name__ == "__main__":
    main()
