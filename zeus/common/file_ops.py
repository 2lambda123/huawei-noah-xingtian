# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FileOps class."""
import os
import pickle
import logging
import shutil

logger = logging.getLogger(__name__)


class FileOps(object):
    """This is a class with some class methods to handle some files or folder."""

    @classmethod
    def make_dir(cls, *args):
        """        Make a new local directory.

        It joins the given list of strings as a new directory path and creates
        the directory if it does not exist.

        Args:
            *args (list of str): List of paths to be joined as a new directory.
        """
        _path = cls.join_path(*args)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def make_base_dir(cls, *args):
        """        Make a new base directory.

        This function takes a list of string paths and joins them to create a
        new base directory. If the directory already exists, it does nothing.

        Args:
            *args (list of str): List of string paths to be joined as a new base directory.

        Returns:
            None: If the directory already exists.
        """
        _file = cls.join_path(*args)
        if os.path.isfile(_file):
            return
        _path, _ = os.path.split(_file)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def join_path(cls, *args):
        """        Join list of path and return.

        This function takes a list of string paths and joins them into a single
        path string. If only one path is provided, it is returned as is. If the
        first path contains a drive letter (e.g., "C:") or a URL (e.g.,
        "http://"), the paths are joined accordingly. If a path starts with a
        slash or a backslash, the leading character is removed before joining.

        Args:
            *args (list of str): List of string paths to be joined.

        Returns:
            str: The joined path string.
        """
        if len(args) == 1:
            return args[0]
        args = list(args)
        for i in range(1, len(args)):
            if args[i][0] in ["/", "\\"]:
                args[i] = args[i][1:]
        # local path
        if ":" not in args[0]:
            args = tuple(args)
            return os.path.join(*args)
        # http or s3 path
        prefix = args[0]
        if prefix[-1] != "/":
            prefix += "/"
        tail = os.path.join(*args[1:])
        return prefix + tail

    @classmethod
    def dump_pickle(cls, obj, filename):
        """        Dump a object to a file using pickle.

        It dumps the given object to a file using the pickle module. If the file
        does not exist, it creates the necessary directories and then writes the
        object to the file.

        Args:
            cls: The class object.
            obj: The target object to be dumped.
            filename: The target pickle file path.
        """
        if not os.path.isfile(filename):
            cls.make_base_dir(filename)
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load_pickle(cls, filename):
        """        Load a pickle file and return the object.

        Args:
            filename (str): Target pickle file path.

        Returns:
            object or None: The loaded original object, or None if the file does not
                exist.
        """
        if not os.path.isfile(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

    @classmethod
    def copy_folder(cls, src, dst):
        """        Copy a folder from source to destination.

        Args:
            src (str): Source path.
            dst (str): Destination path.

        Returns:
            None: If destination path is None or empty.
        """
        if dst is None or dst == "":
            return
        try:
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                else:
                    if not os.path.samefile(src, dst):
                        for files in os.listdir(src):
                            name = os.path.join(src, files)
                            back_name = os.path.join(dst, files)
                            if os.path.isfile(name):
                                shutil.copy(name, back_name)
                            else:
                                if not os.path.isdir(back_name):
                                    shutil.copytree(name, back_name)
                                else:
                                    cls.copy_folder(name, back_name)
            else:
                logger.error("failed to copy folder, folder is not existed, folder={}.".format(src))
        except Exception as ex:
            logger.error("failed to copy folder, src={}, dst={}, msg={}".format(src, dst, str(ex)))

    @classmethod
    def copy_file(cls, src, dst):
        """        Copy a file from source to destination.

        Args:
            cls: The class instance.
            src (str): Source path.
            dst (str): Destination path.
        """
        if dst is None or dst == "":
            return
        try:
            if ":" in src:
                cls.http_download(src, dst)
                return
            if os.path.isfile(src):
                shutil.copy(src, dst)
            else:
                logger.error("failed to copy file, file is not existed, file={}.".format(src))
        except Exception as ex:
            logger.error("Failed to copy file, src={}, dst={}, msg={}".format(src, dst, str(ex)))

    @classmethod
    def download_dataset(cls, src_path, local_path=None):
        """        Download dataset from http or https web site, return path.

        It downloads the dataset from the specified http or https web site and
        returns the final data path. If the local path is not provided, it
        creates a temporary directory to store the downloaded dataset.
        """
        if src_path is None:
            raise FileNotFoundError("Dataset path is None, please set dataset path in config file.")
        if src_path.lower().startswith("http://") or src_path.lower().startswith("https://"):
            if local_path is None:
                local_path = os.path.abspath("./temp")
            cls.make_dir(local_path)
            base_name = os.path.basename(src_path)
            local_path = os.path.join(local_path, base_name)
            logger.debug("Downloading, from={}, to={}.".format(src_path, local_path))
            cls.http_download(src_path, local_path, unzip=True)
            return os.path.dirname(local_path)
        if os.path.exists(src_path):
            return src_path
        else:
            raise FileNotFoundError('Path is not existed, path={}'.format(src_path))

    @classmethod
    def http_download(cls, src, dst, unzip=False):
        """        Download data from http or https web site.
        """
        from six.moves import urllib
        import fcntl

        signal_file = cls.join_path(os.path.dirname(dst), ".{}.signal".format(os.path.basename(dst)))
        if not os.path.isfile(signal_file):
            with open(signal_file, 'w') as fp:
                fp.write('{}'.format(0))

        with open(signal_file, 'r+') as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            signal = int(fp.readline(5_000_000).strip())
            if signal == 0:
                try:
                    urllib.request.urlretrieve(src, dst)
                    logger.info("Downloaded completely.")
                except (urllib.error.URLError, IOError) as e:
                    logger.error("Faild download, msg={}".format(str(e)))
                    raise e
                if unzip is True and dst.endswith(".tar.gz"):
                    logger.info("Untar dataset file, file={}".format(dst))
                    cls._untar(dst)
                    logger.info("Untar dataset file completely.")
                with open(signal_file, 'w') as fn:
                    fn.write('{}'.format(1))
            else:
                logging.debug("File is already downloaded, file={}".format(dst))
            fcntl.flock(fp, fcntl.LOCK_UN)

    @classmethod
    def _untar(cls, src, dst=None):
        """Extracts the contents of a tar file to the specified destination
        directory.

        Args:
            src (str): The path to the tar file.
            dst (str?): The destination directory where the contents will be extracted. If not
                provided, the parent directory of the tar file will be used.
        """

        import tarfile
        if dst is None:
            dst = os.path.dirname(src)
        with tarfile.open(src, 'r:gz') as tar:
            tar.extractall(path=dst)

    @classmethod
    def exists(cls, path):
        """        Check if a folder or file exists at the given path.

        Args:
            path (str): The path to the folder or file.

        Returns:
            bool: True if the folder or file exists, False otherwise.
        """
        return os.path.isdir(path) or os.path.isfile(path)
