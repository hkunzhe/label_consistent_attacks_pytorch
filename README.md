# Label-Consistent Backdoor Attacks

This repository contains the minimal PyTorch implementation of [Label-Consistent Backdoor Attacks](https://arxiv.org/abs/1912.02771) on CIFAR-10 dataset.
The official Tensorflow implementation is [here](https://github.com/MadryLab/label-consistent-backdoor-code).

## Requirements
- python 3.7
- pytorch 1.6.0
- numpy
- tabulate
- pyyaml
- tqdm

## Usage

### Step1: Train an Adversarially Robust Model
For fast adversarial training, please refer to [fast_adversarial](https://github.com/locuslab/fast_adversarial). I also provide a PGD adversarially pretrained ResNet-18 model in [here](). I get the pretrained model by running [train_pgd.py](https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_pgd.py). The parameters of PGD advesarial traning is the same with the paper, which in turn was adapted from [cifar10_challenge](https://github.com/MadryLab/cifar10_challenge).

### Step2: Generate Adversarially Peturbed Dataset
```
python craft_adv_dataset.py --config config/craft.yaml --gpu 0
```
This will generate adversarially peturbed dataset to `data/adv_dataset/craft.npz` by PGD attack. The parameters of PGD attack are the same with the paper.

### Step3: Train a Backdoored Model
```
python poison_train.py --config config/poison_train.yaml --gpu 0
```
__Note:__ For simplicity, I use a [randomly generated trigger]() instead of a less visible and four-corner trigger in the paper section 4.4 (Improving backdoor trigger design). And the parameters of poison training in `config/poison_train.yaml` are adapted from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), which may be different from the paper. Please refer to the experimental setup in the paper Appendix A.