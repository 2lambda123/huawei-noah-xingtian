# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register zeus modules automatically."""

import os
import pickle
import psutil
import logging
import subprocess
import json
import traceback
import copy
import signal
import zeus
import fickling


def run_remote_worker(worker_id, worker_path, id):
    """    Run worker on a remote machine.

    This function initializes logging, loads configuration, registers Zeus,
    and runs the worker on a remote machine.

    Args:
        worker_id (int): The ID of the worker.
        worker_path (str): The path to the worker.
        id (int): The ID.

    Returns:
        int: 0 upon successful completion.
    """
    from zeus.common.utils import init_log
    init_log(level="info",
             log_file=".temp_{}.log".format(worker_id),
             log_path=worker_path)

    config = _load_config(worker_id, worker_path, id)
    zeus.register_zeus(os.environ['BACKEND_TYPE'].lower())

    if zeus.is_gpu_device():
        sub_pid_list = call_in_gpu(config, id, worker_id, worker_path)
    elif zeus.is_npu_device():
        sub_pid_list = call_in_npu(config, id, worker_id, worker_path)
    logging.info("DistributedWorker finished!")
    for sub_pid in sub_pid_list:
        kill_proc_tree(pid=sub_pid)
    logging.info("DistributedWorker subprocess cleaned!")
    return 0


def _load_config(worker_id, worker_path, id):
    """Load configuration data for a specific worker.

    This function loads the configuration data for a specific worker based
    on the worker ID and path.

    Args:
        worker_id (int): The ID of the worker.
        worker_path (str): The path where the worker configuration file is located.
        id (int): The specific ID used to identify the configuration file.

    Returns:
        dict: A dictionary containing the configuration data for the worker.
    """

    _config_file = os.path.join(
        worker_path,
        '.{0}.c.pkl'.format(id))
    with open(_config_file, 'rb') as f:
        config = fickling.load(f)
    return config


def kill_proc_tree(pid, sig=signal.SIGKILL, include_parent=True,
                   timeout=None, on_terminate=None):
    """    Kill a process tree (including grandchildren) with a specified signal.

    This function sends a specified signal to the process tree rooted at the
    given PID. It includes the parent process if specified and provides
    options for timeout and callback upon termination.
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


def call_in_gpu(config, id, worker_id, worker_path):
    """    Call function based on GPU devices.

    This function sets up the environment variables for GPU processing based
    on the provided configuration. It assigns the worker's NCCL port and
    updates the PYTHONPATH accordingly.

    Args:
        config (dict): A dictionary containing configuration parameters.
        id (int): An identifier for the function call.
        worker_id (int): An identifier for the worker.
        worker_path (str): The path to the worker.

    Returns:
        list: A list containing the subprocess PID.
    """
    env = os.environ.copy()
    sub_pid_list = []
    worker_nccl_port = config["worker_nccl_port"]
    world_size = config["world_size"]
    if 'CUDA_VISIBLE_DEVICES' in env:
        try:
            first_gpu_id = env['CUDA_VISIBLE_DEVICES'].split(",")[0]
            env['VEGA_WORKER_PORT'] = '{}'.format(worker_nccl_port + int(first_gpu_id))
        except Exception:
            env['VEGA_WORKER_PORT'] = '{}'.format(worker_nccl_port)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = "{}:{}:{}".format(
            env['PYTHONPATH'], worker_path, os.path.abspath(os.curdir))
    elif worker_id is not None and worker_path is not None:
        env['PYTHONPATH'] = "{}:{}".format(
            worker_path, os.path.abspath(os.curdir))
    sub_pid = _subprocess(config, id, worker_id, worker_path, rank=0, world_size=world_size,
                          env=env, is_backend=False)
    sub_pid_list.append(sub_pid)
    return sub_pid_list


def call_in_npu(config, id, worker_id, worker_path):
    """    Call function based on NPU devices.

    This function is responsible for calling a function based on NPU
    devices. It sets up the environment, reads the rank table file, and
    handles the configuration based on the 'dft' flag in the config.

    Args:
        config (dict): Configuration settings.
        id (int): Identifier.
        worker_id (int): Worker identifier.
        worker_path (str): Path to the worker.

    Returns:
        list: List containing the subprocess PID.
    """
    env = os.environ.copy()
    sub_pid_list = []
    npu_call_path = os.path.join(config["device_folder"], 'npu')
    if not os.path.exists(npu_call_path):
        os.makedirs(npu_call_path, exist_ok=True)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = "{}:{}:{}".format(
            env['PYTHONPATH'], worker_path, os.path.abspath(os.curdir))
    elif worker_id is not None and worker_path is not None:
        env['PYTHONPATH'] = "{}:{}".format(
            worker_path, os.path.abspath(os.curdir))
    rank_file = env.get('RANK_TABLE_FILE')
    with open(rank_file, 'r') as f:
        rank_table_json = json.loads(f.read())
    if config["general"].get('dft', False):
        env['RANK_SIZE'] = env['ORIGIN_RANK_SIZE']
        env['RANK_TABLE_FILE'] = env['ORIGIN_RANK_TABLE_FILE']
    else:
        env['RANK_SIZE'] = '1'
        env['DEVICE_ID'] = rank_table_json['server_list'][0]['device'][0]['device_id']
        env['RANK_ID'] = env['DEVICE_ID']
        env.pop('RANK_TABLE_FILE', None)
    from zeus.common import switch_directory
    with switch_directory(os.path.join(npu_call_path, 'device%s' % str(worker_id))):
        sub_pid = _subprocess(config, id, worker_id, worker_path, rank=0, world_size=1, env=env, is_backend=False)
    sub_pid_list.append(sub_pid)
    return sub_pid_list


def _subprocess(config, id, worker_id, worker_path, rank, world_size, env, is_backend=False):
    """    Subprocess on each rank.

    Load pickle file into worker class, and use subprocess to run the
    train_process function.

    Args:
        config (dict): Configuration settings.
        id (str): Identifier for the subprocess.
        worker_id (str): Identifier for the worker.
        worker_path (str): Path to the worker.
        rank (int): Node rank.
        world_size (int): Number of total nodes.
        env (dict): Environment variables.
        is_backend (bool?): Flag indicating if it's a backend process.

    Returns:
        int: Process ID of the subprocess.
    """
    env['RANK'] = "{}".format(rank)
    env['WORLD_SIZE'] = "{}".format(world_size)

    _refresh_config_file(config, id, worker_id, worker_path, env)

    config_file = os.path.join(
        worker_path,
        '.{0}.c.pkl'.format(id))
    worker_file = os.path.join(
        worker_path,
        '.{0}.w.pkl'.format(id))

    cmd = "from zeus.trainer.deserialize import load_config;"
    cmd += "load_config('{}');".format(config_file)

    if 'VEGA_INIT_ENV' in os.environ:
        cmd += os.environ.copy()['VEGA_INIT_ENV']

    cmd += "from zeus.trainer.deserialize import load_worker;"
    cmd += "worker=load_worker('{}');".format(worker_file)
    cmd += "worker.train_process();"

    if is_backend:
        proc = subprocess.Popen(['python3', '-c', cmd], close_fds=True, env=env)
        pid = proc.pid
    else:
        try:
            proc = subprocess.Popen(['python3', '-c', cmd], env=env)
            pid = proc.pid
            proc.wait(timeout=config["timeout"])
        except Exception:
            logging.warn("Timeout worker has been killed.")
            logging.warn(traceback.print_exc())
    return pid


def _refresh_config_file(config, id, worker_id, worker_path, env):
    """Refresh the configuration file with updated environment variables.

    This function updates the configuration dictionary with the provided
    environment variables. It then saves the updated configuration to a
    pickle file in the worker's path.

    Args:
        config (dict): The configuration dictionary to be updated.
        id (int): The identifier for the configuration.
        worker_id (int): The identifier for the worker.
        worker_path (str): The path to the worker's directory.
        env (dict): A dictionary containing environment variables.
    """

    config["env"]["RANK"] = env.get("RANK", None)
    config["env"]["WORLD_SIZE"] = env.get("WORLD_SIZE", None)
    config["env"]["PYTHONPATH"] = env.get("PYTHONPATH", None)
    config["env"]["RANK_TABLE_FILE"] = env.get("RANK_TABLE_FILE", None)
    config["env"]["RANK_SIZE"] = env.get("RANK_SIZE", None)
    config["env"]["DEVICE_ID"] = env.get("DEVICE_ID", None)
    config["env"]["RANK_ID"] = env.get("RANK_ID", None)

    config_file = os.path.join(
        worker_path,
        '.{0}.c.pkl'.format(id))
    with open(config_file, "wb") as f:
        pickle.dump(config, f)
