import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms

from .randaugment import RandAugmentMC


def train_split(labels, labeled_ratio=0.1):
    labels = np.array(labels)
    n_samples = len(labels)
    rng = np.random.default_rng(0)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    num_labeled = max(1, int(n_samples * labeled_ratio))

    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]

    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, img_size):
        self.weak = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=16),
                transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=3, m=4)])
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        # return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        return self.normalize(weak), self.normalize(strong)


def get_imdbwiki(csv_path, img_dir, img_size, labeled_ratio=0.1, ssl_mult=1):
    transform_labeled = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
    transform_val = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
    
    train_dataset = IMDBWIKI(
        csv_path, img_dir, split='train', transform=None)
    print("total train data:", len(train_dataset))
    
    mean, std = train_dataset.get_mean_std()
    print(f"Mean of targets: {mean:.2f}")
    print(f"Std of targets: {std:.2f}")
    
    train_labeled_idxs, train_unlabeled_idxs = train_split(train_dataset.targets, labeled_ratio)

    train_labeled_dataset = IMDBWIKI_SSL(
        csv_path, img_dir, split='train', index_list=train_labeled_idxs,
        transform=transform_labeled, is_labeled=True, ssl_mult=ssl_mult)
    print("labeled data after duplicates:", len(train_labeled_dataset))

    train_unlabeled_dataset = IMDBWIKI_SSL(
        csv_path, img_dir, split='train', index_list=train_unlabeled_idxs,
        transform=TransformFixMatch(img_size=img_size), is_labeled=False)
    print("Using SSL_SPLIT unlabeled", len(train_unlabeled_dataset))

    val_dataset = IMDBWIKI(
        csv_path, img_dir, split='val', transform=transform_val)
    print("val data", len(val_dataset))

    test_dataset = IMDBWIKI(
        csv_path, img_dir, split='test', transform=transform_val)
    print("test data", len(test_dataset))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


class IMDBWIKI(data.Dataset):
    def __init__(self, csv_path, img_dir, split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path).query(f'SPLIT == "{split}"')
        self.data = df['path'].tolist()
        self.targets = df['age'].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        with Image.open(os.path.join(self.img_dir, self.data[index])) as img:
            img = img.convert('RGB')
        label = np.asarray(self.targets[index]).astype('float32')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def get_mean_std(self):
        targets = np.array(self.targets, dtype=np.float32)
        mean = np.mean(targets)
        std = np.std(targets)
        return mean, std


class IMDBWIKI_SSL(IMDBWIKI):
    def __init__(self, csv_path, img_dir, split='train', transform=None, index_list=None, is_labeled=False, ssl_mult=1):
        super().__init__(csv_path, img_dir, split, transform)
        
        self.ssl_mult = ssl_mult
        if index_list is not None:
            self.data = [self.data[i] for i in index_list]
            self.targets = [self.targets[i] for i in index_list]

        if is_labeled:
            self.data = self.data * self.ssl_mult
            self.targets = self.targets * self.ssl_mult
        

