# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Utils functions that been used in pipeline."""
import os
import socket
import subprocess
import sys
import logging
import signal
import psutil
from collections import OrderedDict
from enum import Enum
from zeus.common import FileOps
from zeus.common.task_ops import TaskOps


class WorkerTypes(Enum):
    """WorkerTypes."""

    TRAINER = 1
    EVALUATOR = 2
    HOST_EVALUATOR = 3
    HAVA_D_EVALUATOR = 4
    DeviceEvaluator = 5


class PairDictQueue():
    """A special Dict Queue only for Master to use to collect all finished Evaluator results.

    the insert and pop item could only be string or int.
    as a example for how to used in Evalutor, the stored odict could be :
    {
        "step_name::worker1": {"EVALUATE_GPU":0, "EVALUATE_DLOOP":0},
        "step_name::worker2": {"EVALUATE_GPU":0, "EVALUATE_DLOOP":1},
        "step_name::worker3": {"EVALUATE_GPU":1, "EVALUATE_DLOOP":0},
        "step_name::worker4": {"EVALUATE_GPU":1, "EVALUATE_DLOOP":1},
    }
    the list could mean each sub-evalutor-worker's status, 0 is not finished,
    1 is finished, here as example, this list could mean [gpu, dloop].
    and the key of odict is the id of this task(which combined with step name
    and worker-id).
    Only sub-evalutor-worker's all status turn to 1(finshed), could it be able
    to be popped from this PairDictQueue.

    :param int pair_size: Description of parameter `pair_size`.
    """

    def __init__(self):
        self.dq_id = 0
        self.odict = OrderedDict()
        return

    def add_new(self, item, type):
        """        Add a new item of a specified type to the dictionary.

        This function adds a new item of a specified type to the dictionary. If
        the item does not exist in the dictionary, it creates a new entry for
        the item and sets the type to the specified value.

        Args:
            item (str): The item to be added to the dictionary.
            type (str): The type of the item to be added.
        """
        if item not in self.odict:
            self.odict[item] = dict()
        self.odict[item][type] = 0

    def put(self, item, type):
        """        Short summary.

        Args:
            item (type): Description of parameter `item`.
            type (type): Description of parameter `type`.

        Returns:
            bool: Description of returned object, True if successful.
        """
        if item not in self.odict:
            logging.debug("item({}) not in PairDictQueue!".format(item))
            return
        self.odict[item][type] = 1
        logging.debug("PairDictQueue add item({}) key({})".format(item, type))
        return True

    def get(self):
        """        Get the first item from the ordered dictionary where all values are 1.

        This method iterates through the ordered dictionary and returns the
        first key where all values are 1.

        Returns:
            Any: The first key from the ordered dictionary where all values are 1, or
                None if no such key is found.
        """
        item = None
        for key, subdict in self.odict.items():
            item_ok = True
            for k, i in subdict.items():
                if i != 1:
                    item_ok = False
                    break
            if item_ok:
                self.odict.pop(key)
                item = key
                break
        return item

    def qsize(self):
        """        Return the number of items in the ordered dictionary.

        Returns:
            int: The number of items in the ordered dictionary.
        """
        return len(self.odict)


# Here start the stand alone functions for master to use!
def clean_cuda_proc(master_pid, device_id):
    """    Clean up CUDA processes associated with a specific device.

    This function kills all CUDA processes associated with the specified
    device, except for the master process and the current process.

    Args:
        master_pid (int): The process ID of the master process.
        device_id (int): The ID of the CUDA device.
    """
    current_pid = os.getpid()
    cuda_kill = "fuser -v /dev/nvidia{0} | " \
                "awk '{{for(i=1;i<=NF;i++)if($i!={1}&&$i!={2})" \
                "print \"kill -9 \" $i;}}' | sh".format(device_id, master_pid, current_pid)
    os.system(cuda_kill)
    return


def kill_children_proc(sig=signal.SIGTERM, recursive=True,
                       timeout=1, on_terminate=None):
    """    Kill a process tree of the current process (including grandchildren).

    This function sends the specified signal "sig" to the current process
    and its children, including grandchildren, and returns a tuple
    containing processes that have terminated and processes that are still
    running. If the "on_terminate" parameter is specified, it is a callback
    function that is called as soon as a child process terminates.

    Args:
        sig (int): The signal to be sent to the process tree (default is signal.SIGTERM).
        recursive (bool): If True, kills all child processes recursively (default is True).
        timeout (int): The timeout for waiting for child processes to terminate (default is 1
            second).
        on_terminate (callable): A callback function to be called when a child process terminates.

    Returns:
        tuple: A tuple containing two lists - the first list contains the processes
            that have terminated, and the second list contains the processes that
            are still running.
    """
    pid = os.getpid()
    parent = psutil.Process(pid)
    children = parent.children(recursive)
    for p in children:
        logging.info("children: {}".format(p.as_dict(attrs=['pid', 'name', 'username'])))
        p.send_signal(sig)
    gone, alive = psutil.wait_procs(children, timeout=timeout,
                                    callback=on_terminate)
    return (gone, alive)


def kill_proc_tree(pid, sig=signal.SIGKILL, include_parent=True,
                   timeout=None, on_terminate=None):
    """    Kill a process tree (including grandchildren) with a specified signal.

    This function sends the specified signal to the process tree rooted at
    the given process ID. If 'include_parent' is True, the signal is also
    sent to the parent process. The 'timeout' parameter can be used to
    specify a timeout for waiting for the processes to terminate. The
    'on_terminate' parameter, if specified, is a callback function which is
    called as soon as a child terminates.
    """
    if pid == os.getpid():
        raise RuntimeError("I refuse to kill myself")
    gone = None
    alive = None
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        if include_parent:
            children.append(parent)
        for p in children:
            p.send_signal(sig)
        gone, alive = psutil.wait_procs(children, timeout=timeout,
                                        callback=on_terminate)
    except Exception:

        pass
    return (gone, alive)


def install_and_import_local(package, package_path=None, update=False):
    """    Install and import local python packages.

    This function installs and imports a local python package. If the
    package is not already installed, it will be installed using pip.

    Args:
        package (str): The name of the package to install and import.
        package_path (str?): The path to the local wheel file of the package. Defaults to None.
        update (bool): If True, the function will update the package if it is already
            installed. Defaults to False.
    """
    import importlib
    try:
        if not update:
            try:
                importlib.import_module(package)
            except ImportError:
                import pip
                if hasattr(pip, 'main'):
                    pip.main(['install', package_path])
                elif hasattr(pip, '_internal'):
                    pip._internal.main(['install', package_path])
                else:
                    subprocess.call([sys.executable, "-m", "pip", "install",
                                     package_path])
        else:
            import pip
            if hasattr(pip, 'main'):
                pip.main(['install', '-U', package_path])
            elif hasattr(pip, '_internal'):
                pip._internal.main(['install', '-U', package_path])
            else:
                subprocess.call([sys.executable, "-m", "pip", "install", "-U",
                                 package_path])
    finally:
        globals()[package] = importlib.import_module(package)


def get_master_address(args):
    """    Get master address(ip, port) from `args.init_method`.

    This function extracts the master address (ip, port) from the
    `args.init_method` and returns it.

    Args:
        args (argparse.ArgumentParser): An argparse object containing `init_method`, `rank`, and `world_size`.

    Returns:
        tuple: A tuple containing the IP address (str) and port (str) of the master, or
            None if `args.init_method` is None.
    """
    if args.init_method is not None:
        address = args.init_method[6:].split(":")
        ip = socket.gethostbyname(address[0])
        port = address[-1]
        logging.info("get master address, address={}, ip={}, port={}".format(
            address, ip, port
        ))
        return ip, port
    else:
        logging.warn("fail to get master address, args.init_method is none.")
        return None


def get_local_address():
    """    Try to get the local node's IP.

    This function attempts to retrieve the local node's IP address using the
    socket module. It first gets the hostname and then retrieves the IP
    address associated with that hostname.

    Returns:
        str: The IP address of the local node.
    """
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    logging.info("get local address, hostname={}, ip={}".format(
        hostname, ip
    ))
    return ip


def save_master_ip(ip_address, port, args):
    """    Write the ip and port in a system path.

    Args:
        ip_address (str): The IP address to be written.
        port (str): The port to be written.
        args (argparse.ArgumentParser): An argparse object that should contain
            `init_method`, `rank`, and `world_size`.
    """
    temp_folder = TaskOps().temp_path
    FileOps.make_dir(temp_folder)
    file_path = os.path.join(temp_folder, 'ip_address.txt')
    logging.info("write ip, file path={}".format(file_path))
    with open(file_path, 'w') as f:
        f.write(ip_address + "\n")
        f.write(port + "\n")


def load_master_ip():
    """    Get the ip and port that are written in a system path.

    This function retrieves the ip and port from a file stored in the system
    path without downloading anything from S3.

    Returns:
        tuple: A tuple containing the ip and port retrieved from the file.
    """
    temp_folder = TaskOps().temp_path
    FileOps.make_dir(temp_folder)
    file_path = os.path.join(temp_folder, 'ip_address.txt')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            ip = f.readline(5_000_000).strip()
            port = f.readline(5_000_000).strip()
            logging.info("get write ip, ip={}, port={}".format(
                ip, port
            ))
            return ip, port
    else:
        return None, None


def get_master_port(args):
    """    Get master port from `args.init_method`.

    This function extracts the port used by the master to communicate with
    slaves from the `args.init_method` argument.

    Args:
        args (argparse.ArgumentParser): An argparse object that should contain `init_method`, `rank`, and
            `world_size`.

    Returns:
        str or None: The port that the master used to communicate with slaves,
            or None if `args.init_method` is not provided.
    """
    if args.init_method is not None:
        address = args.init_method.split(":")
        port = address[-1]
        return port
    else:
        return None
