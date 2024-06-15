# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class of Cityscapes dataset."""
import os.path as osp
import cv2
import numpy as np
import glob
from .utils.dataset import Dataset
from zeus.common import ClassFactory, ClassType
from zeus.common import FileOps
from zeus.datasets.conf.city_scapes import CityscapesConfig
import fickling


@ClassFactory.register(ClassType.DATASET)
class Cityscapes(Dataset):
    """Class of Cityscapes dataset, which is subclass of Dateset.

    Two types of data are supported:
        1) Image with extensions in 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP'
        2) pkl with extensions in 'pkl', 'pt', 'pth'. Image pkl should be in format of HWC, with bgr as the channels
    To use this dataset, provide either: 1) data_dir and label_dir; or 2) root_dir and list_file
    :param train: if the mdoe is train or false, defaults to True
    :type train: bool, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = CityscapesConfig()

    def __init__(self, **kwargs):
        """Construct the Cityscapes class."""
        super(Cityscapes, self).__init__(**kwargs)
        self.dataset_init()

    def _init_transforms(self):
        """        Initialize transforms.

        This function initializes a list of transforms based on the provided
        arguments. It checks for specific transform types in the arguments and
        creates instances of those transforms.

        Returns:
            list: A list of initialized transform instances.
        """
        result = list()
        if "Rescale" in self.args:
            import logging
            logging.info(str(dict(**self.args.Rescale)))
            result.append(self._get_cls("Rescale_pair")(**self.args.Rescale))
        if "RandomMirror" in self.args and self.args.RandomMirror:
            result.append(self._get_cls("RandomHorizontalFlip_pair")())
        if "RandomColor" in self.args:
            result.append(self._get_cls("RandomColor_pair")(**self.args.RandomColor))
        if "RandomGaussianBlur" in self.args:
            result.append(self._get_cls("RandomGaussianBlur_pair")(**self.args.RandomGaussianBlur))
        if "RandomRotation" in self.args:
            result.append(self._get_cls("RandomRotate_pair")(**self.args.RandomRotation))
        if "Normalization" in self.args:
            result.append(self._get_cls("Normalize_pair")(**self.args.Normalization))
        if "RandomCrop" in self.args:
            result.append(self._get_cls("RandomCrop_pair")(**self.args.RandomCrop))
        return result

    def _get_cls(self, _name):
        """Get a class object based on the provided name for transformation.

        This function retrieves a class object from the ClassFactory for
        transformation based on the given name.

        Args:
            _name (str): The name of the class to retrieve.

        Returns:
            class: The class object for transformation.
        """

        return ClassFactory.get_cls(ClassType.TRANSFORM, _name)

    def dataset_init(self):
        """        Construct method.

        If both data_dir and label_dir are provided, then use data_dir and
        label_dir. Otherwise, use root_dir and list_file.
        """
        if "data_dir" in self.args and "label_dir" in self.args:
            self.args.data_dir = FileOps.download_dataset(self.args.data_dir)
            self.args.label_dir = FileOps.download_dataset(self.args.label_dir)
            self.data_files = sorted(glob.glob(osp.join(self.args.data_dir, "*")))
            self.label_files = sorted(glob.glob(osp.join(self.args.label_dir, "*")))
        else:
            if "root_dir" not in self.args or "list_file" not in self.args:
                raise Exception("You must provide a root_dir and a list_file!")
            self.args.root_dir = FileOps.download_dataset(self.args.root_dir)
            with open(osp.join(self.args.root_dir, self.args.list_file)) as f:
                lines = f.readlines()
            self.data_files = [None] * len(lines)
            self.label_files = [None] * len(lines)
            for i, line in enumerate(lines):
                data_file_name, label_file_name = line.strip().split()
                self.data_files[i] = osp.join(self.args.root_dir, data_file_name)
                self.label_files[i] = osp.join(self.args.root_dir, label_file_name)

        datatype = self._get_datatype()
        if datatype == "image":
            self.read_fn = self._read_item_image
        else:
            self.read_fn = self._read_item_pickle

    def __len__(self):
        """        Get the length of the dataset.

        This method returns the length of the dataset by counting the number of
        data files present.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_files)

    def __getitem__(self, index):
        """        Get an item of the dataset according to the index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing image and mask.
                - image (numpy.ndarray): The image data.
                - mask (numpy.ndarray): The mask data.
        """
        image, label = self.read_fn(index)
        image_name = self.data_files[index].split("/")[-1].split(".")[0]
        image, label = self.transforms(image, label)
        image = np.transpose(image, [2, 0, 1]).astype(np.float32)
        mask = label.astype(np.int64)

        return image, mask

    @staticmethod
    def _get_datatype_files(file_paths):
        """        Check file extensions in file_paths to decide whether they are images or
        pkl.
        """
        IMG_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG',
                          'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP'}
        PKL_EXTENSIONS = {'pkl', 'pt', 'pth'}

        file_extensions = set(data_file.split('.')[-1] for data_file in file_paths)
        if file_extensions.issubset(IMG_EXTENSIONS):
            return "image"
        elif file_extensions.issubset(PKL_EXTENSIONS):
            return "pkl"
        else:
            raise Exception("Invalid file extension")

    def _get_datatype(self):
        """        Check the datatype of all data.

        This function checks the datatype of all data files and label files
        provided. It compares the datatype of data files with label files and
        ensures they are of the same type.
        """
        type_data = self._get_datatype_files(self.data_files)
        type_labels = self._get_datatype_files(self.label_files)

        if type_data == type_labels:
            return type_data
        else:
            raise Exception("Images and masks must be both image or pkl!")

    def _read_item_image(self, index):
        """        Read image and label in "image" format.

        This function reads an image and its corresponding label in "image"
        format. It reads the image and label files using OpenCV and returns them
        as numpy arrays.

        Args:
            index (int): The index of the item to read.

        Returns:
            tuple of np.array: A tuple containing the image as a numpy array in HWC
                format (bgr)
            and the label as a numpy array in HW format.
        """
        image = cv2.imread(self.data_files[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_files[index], cv2.IMREAD_GRAYSCALE)
        return image, label

    def _read_item_pickle(self, index):
        """        Read image and label in "pkl" format.

        This function reads an image and label stored in "pkl" format.

        Args:
            index (int): Index of the item to read.

        Returns:
            tuple of np.array: A tuple containing the image in np.array format (HWC,
                bgr) and the label in np.array format (HW).
        """
        with open(self.data_files[index], "rb") as file:
            image = fickling.load(file)
        with open(self.label_files[index], "rb") as file:
            label = fickling.load(file)
        return image, label

    @property
    def input_size(self):
        """        Get the input size of Cityspace.

        This function returns the size of the input data along the second
        dimension.

        Returns:
            int: The size of the input data along the second dimension.
        """
        _shape = self.data.shape
        return _shape[1]
