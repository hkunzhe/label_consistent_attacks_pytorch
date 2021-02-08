import copy

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class CleanLabelDataset(Dataset):
    """Clean-label dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        adv_dataset_path (str): The adversarially perturbed dataset path.
        transform (callable): The backdoor transformations.
        poison_idx (np.array): An 0/1 (clean/poisoned) array with
            shape `(len(dataset), )`.
        target_label (int): The target label.
    """

    def __init__(self, dataset, adv_dataset_path, transform, poison_idx, target_label):
        super(CleanLabelDataset, self).__init__()
        self.clean_dataset = copy.deepcopy(dataset)
        self.adv_data = np.load(adv_dataset_path)["data"]
        self.clean_data = self.clean_dataset.data
        self.train = self.clean_dataset.train
        if self.train:
            self.data = np.where(
                (poison_idx == 1)[..., None, None, None],
                self.adv_data,
                self.clean_data,
            )
            self.targets = self.clean_dataset.targets
            self.poison_idx = poison_idx
        else:
            # Only fetch poison data when testing.
            self.data = self.clean_data[np.nonzero(poison_idx)[0]]
            self.targets = self.clean_dataset.targets[np.nonzero(poison_idx)[0]]
            self.poison_idx = poison_idx[poison_idx == 1]
        self.transform = self.clean_dataset.transform
        self.bd_transform = transform
        self.target_label = target_label

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        if self.poison_idx[index] == 1:
            img = self.augment(img, bd_transform=self.bd_transform)
            # If `self.train` is `True`, it will not modify `target` for poison data
            # only in the target class; If `self.train` is `False`, it will flip `target`
            # to `self.target_label` for testing purpose.
            target = self.target_label
        else:
            img = self.augment(img, bd_transform=None)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return len(self.data)

    def augment(self, img, bd_transform=None):
        if bd_transform is not None:
            img = bd_transform(img)
        img = Image.fromarray(img)
        img = self.transform(img)

        return img
