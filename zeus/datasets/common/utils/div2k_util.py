# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This script contains some common function to process the DIV2K dataset."""
import os
import cv2
# import lmdb
import numpy as np
import glob
import fickling

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
PKL_EXTENSIONS = ['.pkl', '.pt', '.pth']


def is_image_file(filename):
    """    To judge whether a given file name is an image or not.

    Args:
        filename (str): The input filename.

    Returns:
        bool: True if the filename is an image, False otherwise.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# image/pkl files will always in the same folder; otherwise use subfile
def get_paths_from_dir(path):
    """    Get all the files in the given directory.

    This function takes a directory path as input and returns a list of file
    names along with their paths.

    Args:
        path (str): The path of the directory.

    Returns:
        list: A list containing the file names and their paths.
    """
    return sorted(glob.glob(os.path.join(path, "*")))


def get_files_datatype(file_names):
    """    Get the datatype of the file.

    This function takes a list of file names and determines the datatype of
    the files based on their extensions.
    """
    extensions = {'.' + file_name.split('.')[-1] for file_name in file_names}
    if extensions.issubset(set(IMG_EXTENSIONS)):
        return 'img'
    elif extensions.issubset(set(PKL_EXTENSIONS)):
        return 'pkl'
    else:
        raise NotImplementedError('Datatype not recognized!')


def get_datatype(dataroot):
    """    Get the datatype of the data path.

    This function takes a data path as input and determines the datatype of
    the data based on the path.

    Args:
        dataroot (str): The data path for which the datatype needs to be determined.

    Returns:
        str: The datatype of the data path.
    """
    if dataroot.endswith(".lmdb"):
        return "lmdb"
    file_names = os.listdir(dataroot)
    return get_files_datatype(file_names)


def read_img_pkl(path):
    """    Read an image from a pickle file.

    This function reads an image from a pickle file located at the specified
    path.

    Args:
        path (str): The file path to the pickle file containing the image.

    Returns:
        tuple: A tuple containing the image data.
    """
    with open(path, "rb") as file:
        return fickling.load(file)


def read_img_img(path):
    """    Read the picture format image.

    This function reads an image file from the specified path using OpenCV
    and returns the image data as a NumPy array.

    Args:
        path (str): The path to the image file.

    Returns:
        ndarray: The image data read from the file.
    """
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def np_to_tensor(np_array):
    """    Convert an image from np array to tensor.

    This function takes an image in the form of a NumPy array with shape HWC
    (Height x Width x Channels) and BGR color format, converts it to a
    PyTorch tensor with shape CHW (Channels x Height x Width) and BGR color
    format.

    Args:
        np_array (np.array): Image in NumPy array format with data type np.uint8 and shape HWC.

    Returns:
        tensor: Image tensor in PyTorch format with data type torch.float32 and shape
            CHW.
    """
    np_array = np.asarray(np_array)
    return np.transpose(np_array, (2, 0, 1)).astype(np.float32)
