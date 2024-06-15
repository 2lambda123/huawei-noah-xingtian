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
import fickling

logger = logging.getLogger(__name__)


class FileOps(object):
    """This is a class with some class methods to handle some files or folder."""

    @classmethod
    def make_dir(cls, *args):
        """        Make a new local directory.

        This function takes a list of string paths and joins them to create a
        new directory. If the directory does not exist, it creates the
        directory.

        Args:
            *args (list of str): List of string paths to be joined as a new directory.
        """
        _path = cls.join_path(*args)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def make_base_dir(cls, *args):
        """        Make a new base directory.

        This function takes a list of string paths and joins them to create a
        new base directory. If the base directory already exists, it does
        nothing. If the base directory does not exist, it creates the directory
        along with any necessary parent directories.

        Args:
            *args (list of str): List of string paths to be joined as a new base directory.
        """
        _file = cls.join_path(*args)
        if os.path.isfile(_file):
            return
        _path, _ = os.path.split(_file)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def join_path(cls, *args):
        """        Join list of paths and return the joined path.

        This function takes a variable number of string paths and joins them
        together to form a single path. If only one path is provided, it is
        returned as is. If multiple paths are provided, they are joined
        together. If the first path is a local path, it is joined using
        os.path.join(). If the first path is a URL or S3 path, the paths are
        concatenated with a '/' separator.

        Args:
            *args (str): Variable number of string paths to be joined.

        Returns:
            str: The joined path as a string.
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
        """        Dump an object to a file using pickle.

        This function takes an object and a file path as input, checks if the
        file exists, creates the base directory if it doesn't, and then dumps
        the object to the file using pickle.

        Args:
            cls: The class instance.
            obj: The object to be dumped.
            filename: The path to the pickle file.
        """
        if not os.path.isfile(filename):
            cls.make_base_dir(filename)
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load_pickle(cls, filename):
        """        Load a pickle file and return the object.

        This function loads a pickle file from the specified path and returns
        the original object.

        Args:
            filename (str): The path to the target pickle file.

        Returns:
            object or None: The loaded original object, or None if the file does not
                exist.
        """
        if not os.path.isfile(filename):
            return None
        with open(filename, "rb") as f:
            return fickling.load(f)

    @classmethod
    def copy_folder(cls, src, dst):
        """        Copy a folder from source to destination.

        This function copies a folder from the source path to the destination
        path. If the destination path is empty or None, the function returns
        without performing any action. If the source is a directory and the
        destination does not exist, it copies the entire directory recursively.
        If the source is a file, it copies the file to the destination. If the
        source is a directory and the destination already exists, it recursively
        copies the contents of the source directory to the destination
        directory.

        Args:
            src (str): The path of the source folder.
            dst (str): The path of the destination folder.
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

        This function copies a file from the source path to the destination
        path. If the destination path is empty or None, the function returns
        without performing any action. If the source path contains a colon
        (':'), it uses the http_download method to download the file. If the
        source path is an existing file, it copies the file using shutil.copy.
        If the source path is not an existing file, it logs an error message.

        Args:
            cls: The class instance.
            src (str): The path of the source file.
            dst (str): The path of the destination file.
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
        """        Download dataset from http or https web site and return the final data
        path.

        This function downloads a dataset from a given http or https web site
        and saves it to the local path. If the source path starts with http://
        or https://, it downloads the dataset to the specified local path. If
        the source path is a local file path, it returns the same path.
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
        """        Download data from an HTTP or HTTPS website.
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
        """Extracts the contents of a tar file to a specified destination
        directory.

        If no destination directory is provided, the files are extracted to the
        directory containing the tar file.

        Args:
            src (str): Path to the tar file to be extracted.
            dst (str?): Destination directory where the contents will be extracted. Defaults to
                None.
        """

        import tarfile
        if dst is None:
            dst = os.path.dirname(src)
        with tarfile.open(src, 'r:gz') as tar:
            tar.extractall(path=dst)

    @classmethod
    def exists(cls, path):
        """        Check if a folder or file exists.

        This function checks whether the specified path corresponds to an
        existing folder or file.

        Args:
            path (str): The path to the folder or file.

        Returns:
            bool: True if the folder or file exists, False otherwise.
        """
        return os.path.isdir(path) or os.path.isfile(path)
