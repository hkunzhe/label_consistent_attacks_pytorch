import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.backdoor import CLBD
from data.cifar import CIFAR10
from data.dataset import CleanLabelDataset
from model.network.resnet import resnet18
from utils import load_config, gen_poison_idx
from trainer import poison_train, test

torch.backends.cudnn.benchmark = True


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/poison_train.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()
    config, _, _ = load_config(args.config)

    print("===Prepare data===")
    bd_config = config["backdoor"]
    print("Load backdoor config:\n{}".format(bd_config))
    bd_transform = CLBD(bd_config["clbd"]["trigger_path"])
    target_label = bd_config["target_label"]
    poison_ratio = bd_config["poison_ratio"]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    print("Load dataset from: {}".format(config["dataset_dir"]))
    clean_train_data = CIFAR10(config["dataset_dir"], train_transform, train=True)
    poison_train_idx = gen_poison_idx(
        clean_train_data, target_label, poison_ratio=poison_ratio
    )
    print(
        "Load the adversarially perturbed dataset from: {}".format(
            config["adv_dataset_path"]
        )
    )
    poison_train_data = CleanLabelDataset(
        clean_train_data,
        config["adv_dataset_path"],
        bd_transform,
        poison_train_idx,
        target_label,
    )
    poison_train_loader = DataLoader(
        poison_train_data, **config["loader"], shuffle=True
    )
    clean_test_data = CIFAR10(config["dataset_dir"], test_transform, train=False)
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = CleanLabelDataset(
        clean_test_data,
        config["adv_dataset_path"],
        bd_transform,
        poison_test_idx,
        target_label,
    )
    clean_test_loader = DataLoader(clean_test_data, **config["loader"])
    poison_test_loader = DataLoader(poison_test_data, **config["loader"])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu = torch.cuda.current_device()
    print("Set gpu to: {}".format(args.gpu))

    model = resnet18()
    model = model.cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), **config["optimizer"]["SGD"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, **config["lr_scheduler"]["multi_step"]
    )

    for epoch in range(config["num_epochs"]):
        print("===Epoch: {}/{}===".format(epoch + 1, config["num_epochs"]))
        print("Poison training...")
        poison_train(model, poison_train_loader, criterion, optimizer)
        print("Test model on clean data...")
        test(model, clean_test_loader, criterion)
        print("Test model on poison data...")
        test(model, poison_test_loader, criterion)

        scheduler.step()
        print("Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"]))


if __name__ == "__main__":
    main()
