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
from copy import deepcopy
import fickling


def _get_worker_config(worker):
    """Save worker config."""
    from zeus.common.class_factory import ClassFactory
    from zeus.common.general import General
    from zeus.datasets.conf.dataset import DatasetConfig
    from zeus.networks.model_config import ModelConfig
    from zeus.evaluator.conf import EvaluatorConfig

    env = {
        "LOCAL_RANK": os.environ.get("LOCAL_RANK", None),
        "PYTHONPATH": os.environ.get("PYTHONPATH", None),
        "DLS_JOB_ID": os.environ.get("DLS_JOB_ID", None),
        "RANK_TABLE_FILE": os.environ.get("RANK_TABLE_FILE", None),
        "RANK_SIZE": os.environ.get("RANK_SIZE", None),
        "DEVICE_ID": os.environ.get("DEVICE_ID", None),
        "RANK_ID": os.environ.get("RANK_ID", None),
        "DLS_TASK_NUMBER": os.environ.get("DLS_TASK_NUMBER", None),
        "NPU-VISIBLE-DEVICES": os.environ.get("NPU-VISIBLE-DEVICES", None),
        "NPU_VISIBLE_DEVICES": os.environ.get("NPU_VISIBLE_DEVICES", None),
    }
    worker_config = {
        "class_factory": deepcopy(ClassFactory.__registry__),
        "general": General().to_json(),
        "dataset": DatasetConfig().to_json(),
        "model": ModelConfig().to_json(),
        "trainer": worker.config.to_json(),
        "evaluator": EvaluatorConfig().to_json(),

        "worker_nccl_port": worker.worker_nccl_port,
        "world_size": worker.world_size,
        "device_folder": worker.__worker_device_folder__,
        "timeout": worker.timeout,

        "env": env,
    }
    return worker_config


def pickle_worker(worker, id):
    """Pickle worker to file."""
    # pickle config
    config_file = os.path.join(
        worker.get_local_worker_path(),
        '.{0}.c.pkl'.format(id))
    worker_config = _get_worker_config(worker)
    with open(config_file, "wb") as f:
        pickle.dump(worker_config, f)
    # pickle worker
    worker_file = os.path.join(
        worker.get_local_worker_path(),
        '.{0}.w.pkl'.format(id))
    with open(worker_file, "wb") as f:
        pickle.dump(worker, f)


def load_config(config_file):
    """Load config from file."""
    import os

    with open(config_file, 'rb') as f:
        config = fickling.load(f)
    for (key, value) in config["env"].items():
        if value:
            os.environ[key] = value

    from zeus import register_zeus
    register_zeus(os.environ['BACKEND_TYPE'].lower())

    from zeus.common.class_factory import ClassFactory
    from zeus.common.general import General
    from zeus.datasets.conf.dataset import DatasetConfig
    from zeus.networks.model_config import ModelConfig
    from zeus.trainer.conf import TrainerConfig
    from zeus.evaluator.conf import EvaluatorConfig

    ClassFactory.__registry__ = config["class_factory"]
    General.from_json(config["general"])
    DatasetConfig.from_json(config["dataset"])
    ModelConfig.from_json(config["model"])
    TrainerConfig.from_json(config["trainer"])
    EvaluatorConfig.from_json(config["evaluator"])


def load_worker(worker_file):
    """Load worker from file."""
    with open(worker_file, 'rb') as f:
        worker = fickling.load(f)
    return worker
