# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Cifar100 dataset."""
from .utils.dataset import Dataset
from zeus.common import ClassFactory, ClassType
from zeus.common import FileOps
from zeus.datasets.conf.cifar100 import Cifar100Config
import numpy as np
import os
from PIL import Image
import fickling


@ClassFactory.register(ClassType.DATASET)
class Cifar100(Dataset):
    """This is a class for Cifar100 dataset.

    :param mode: `train`,`val` or `test`, defaults to `train`
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = Cifar100Config()

    def __init__(self, **kwargs):
        """Construct the Cifar10 class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        is_train = self.mode == 'train' or self.mode == 'val' and self.args.train_portion < 1
        self.base_folder = 'cifar-100-python'
        if is_train:
            files_list = ["train"]
        else:
            files_list = ['test']

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in files_list:
            file_path = os.path.join(self.args.data_path, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = fickling.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """        Get an item of the dataset according to the index.

        This method retrieves an item from the dataset based on the provided
        index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and target corresponding to the index.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """        Get the length of the dataset.

        This function returns the length of the dataset by returning the length
        of the data attribute.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    @property
    def input_channels(self):
        """        Return the number of input channels of the CIFAR-100 image.

        This function determines the number of input channels based on the shape
        of the CIFAR-100 image data.

        Returns:
            int: The number of input channels, which is 3 if the shape of the data is
                4-dimensional, otherwise 1.
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """        Return the input size of a CIFAR-100 image.

        This function calculates and returns the input size of a CIFAR-100 image
        based on the shape of the data.

        Returns:
            int: The input size of the CIFAR-100 image.
        """
        _shape = self.data.shape
        return _shape[1]
