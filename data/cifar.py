import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class CIFAR10(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.transform = transform
        if train:
            data_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            data_list = ["test_batch"]
        data = []
        targets = []
        if root[0] == "~":
            # interprete `~` as the home directory.
            root = os.path.expanduser(root)
        for file_name in data_list:
            file_path = os.path.join(root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])
        # Convert data (List) to NHWC (np.ndarray) works with PIL Image.
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.data = data
        self.targets = np.asarray(targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return len(self.data)
