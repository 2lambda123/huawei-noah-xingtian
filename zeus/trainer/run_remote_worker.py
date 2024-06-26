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
    """Run worker on remote mochine."""
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
    _config_file = os.path.join(
        worker_path,
        '.{0}.c.pkl'.format(id))
    with open(_config_file, 'rb') as f:
        config = fickling.load(f)
    return config


def kill_proc_tree(pid, sig=signal.SIGKILL, include_parent=True,
                   timeout=None, on_terminate=None):
    """Kill a process tree (including grandchildren) with signal.

    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
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


def call_in_gpu(config, id, worker_id, worker_path):
    """Call function based on GPU devices."""
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
    """Call function based on NPU devices."""
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
    """Subprocess on each rank.

    Load pickle file into worker class, and use subprocess to run the
    train_process function.

    :param rank: node rank
    :type rank: int
    :param world_size: number of total nodes
    :type world_size: int
    :param env: environ
    :type env: dict
    :param is_backend: backend or not
    :type is_backend: bool
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
