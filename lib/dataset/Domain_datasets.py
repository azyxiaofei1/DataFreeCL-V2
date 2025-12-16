import json

import numpy as np
import torchvision
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import transforms
from torchvision import datasets
from lib.data_transform.data_transform import AVAILABLE_TRANSFORMS, DATASET_CONFIGS, AVAILABLE_DATASETS
from lib.dataset import SubDataset, split_dataset, TransformedDataset


class Rotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg_min=0, deg_max=180):
        """
        Initializes the rotation with a random angle.
        :param deg_min: lower extreme of the possible random angle
        :param deg_max: upper extreme of the possible random angle
        """
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)

    def __call__(self, x):
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        return torchvision.transforms.functional.rotate(x, self.degrees)


class FixedRotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg):
        """
        Initializes the rotation with a random angle.
        :param deg_min: lower extreme of the possible random angle
        :param deg_max: upper extreme of the possible random angle
        """
        self.degree = deg

    def __call__(self, x):
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        return torchvision.transforms.functional.rotate(x, self.degree)


class Rotate_DomainDataset(Dataset):
    def __init__(self, cfg):
        self.dataset_name = cfg.DATASET.dataset_name
        self.dataset_root = cfg.DATASET.data_root
        self.all_classes = cfg.DATASET.all_classes
        self.all_tasks = cfg.DATASET.all_tasks
        self.classes_per_task = int(self.all_classes / self.all_tasks)
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE
        self.original_imgs_train_datasets = None
        self.val_datasets = None
        self.test_datasets = None
        self.target_transform = None
        self.domain_id = None
        self.val_test_dataset_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_Non_rotated_dataset(self, domain_id):
        self.domain_id = domain_id
        self.original_imgs_train_datasets = datasets.MNIST(root=self.dataset_root, transform=None, download=True)
        self.val_test_dataset_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.val_datasets = datasets.MNIST(root=self.dataset_root, train=False,
                                           transform=self.val_test_dataset_transform,
                                           download=True)

    def get_rotated_dataset(self, domain_id, rotate_angle):
        self.domain_id = domain_id
        self.original_imgs_train_datasets = datasets.MNIST(root=self.dataset_root, transform=transforms.Compose([
            FixedRotation(rotate_angle)
        ]), download=True)
        self.val_test_dataset_transform = transforms.Compose([
            FixedRotation(rotate_angle),
            transforms.ToTensor(),
        ])
        self.val_datasets = datasets.MNIST(root=self.dataset_root, train=False,
                                           transform=self.val_test_dataset_transform,
                                           download=True)

    pass