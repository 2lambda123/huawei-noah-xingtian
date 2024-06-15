# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""
import json
import logging
import os
import glob
import traceback
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import pareto
import time
from threading import Thread
from collections import OrderedDict
from zeus.common import FileOps, TaskOps
from zeus.common.general import General
from zeus.report.share_memory import ShareMemory, LocalShareMemory, ShareMemoryClient
from .nsga_iii import SortAndSelectPopulation
from .record import ReportRecord
from zeus.common.utils import singleton
import fickling


class ReportServer(object):
    """Report class to save all records and broadcast records to share memory."""

    _hist_records = OrderedDict()
    __instances__ = None
    __variables__ = set()

    def __new__(cls, *args, **kwargs):
        """        Override new method to implement a singleton pattern.

        This method overrides the default __new__ method to ensure that only one
        instance of the class is created. If no instances exist, it creates a
        new instance and starts a monitor thread.

        Args:
            cls (class): The class to create a singleton instance for.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The singleton instance of the class.
        """
        if not cls.__instances__:
            cls.__instances__ = super().__new__(cls, *args, **kwargs)
            cls._thread_runing = True
            cls._thread = cls._run_monitor_thread()
        return cls.__instances__

    def add(self, record):
        """        Add a record to the set of historical records.

        This method adds a record to the set of historical records stored in the
        instance.

        Args:
            record (object): The record object to be added.
        """
        self._hist_records[record.uid] = record

    @classmethod
    def add_watched_var(cls, step_name, worker_id):
        """        Add a watched variable to the ReportServer.

        This function adds a variable to the ReportServer's watched variables
        list.

        Args:
            cls (class): The class object.
            step_name (str): The name of the step.
            worker_id (int): The ID of the worker.
        """
        cls.__variables__.add("{}.{}".format(step_name, worker_id))

    @classmethod
    def remove_watched_var(cls, step_name, worker_id):
        """        Remove a watched variable from the ReportServer.

        This function removes a watched variable identified by the combination
        of step_name and worker_id from the ReportServer's internal variables
        list.

        Args:
            cls (object): The class instance representing the ReportServer.
            step_name (str): The name of the step associated with the watched variable.
            worker_id (int): The ID of the worker associated with the watched variable.
        """
        key = "{}.{}".format(step_name, worker_id)
        if key in cls.__variables__:
            cls.__variables__.remove(key)

    @classmethod
    def stop(cls):
        """        Stop the report server.

        This function stops the report server by setting the internal flag
        '_thread_running' to False, joining the thread, and closing the
        ShareMemoryClient.

        Args:
            cls: The class instance of ReportServer.
        """
        if hasattr(ReportServer, "_thread_runing") and ReportServer._thread_runing:
            ReportServer._thread_runing = False
            ReportServer._thread.join()
            ShareMemoryClient().close()

    @classmethod
    def renew(cls):
        """        Renew the report server.

        This function renews the report server by starting a new monitor thread
        if it is not already running.

        Args:
            cls: The class object representing the report server.
        """
        if not hasattr(ReportServer, "_thread_runing") or not ReportServer._thread_runing:
            ReportServer._thread_runing = True
            ReportServer._thread = ReportServer._run_monitor_thread()

    @property
    def all_records(self):
        """        Get all records.

        Returns:
            list: A deep copy of all records stored in the history records dictionary.
        """
        return deepcopy(list(self._hist_records.values()))

    def print_best(self, step_name):
        """        Print the best performance and description for a given step.

        Args:
            step_name (str): The name of the step for which to retrieve the best performance.

        Returns:
            list: A list of dictionaries containing worker_id, performance, and desc for
                the best records.
        """
        records = self.get_pareto_front_records(step_name)
        return [dict(worker_id=record.worker_id, performance=record._performance, desc=record.desc) for record in
                records]

    def pareto_front(self, step_name=None, nums=None, records=None):
        """        Get the Pareto front for a given step name and records.

        This function calculates the Pareto front based on the rewards of the
        records that match the provided step name. If no records are provided,
        it uses all available records. It then applies Pareto sorting to select
        the non-dominated solutions based on the rewards.

        Args:
            step_name (str): The name of the step to filter the records.
            nums (int): The number of solutions to select in the Pareto front.
            records (list): A list of records to consider for Pareto front calculation.

        Returns:
            tuple: A tuple containing two lists:
                - List of selected solutions in the Pareto front.
                - List of indices of the selected solutions in the original list.
        """
        if records is None:
            records = self.all_records
            records = list(filter(lambda x: x.step_name == step_name and x.performance is not None, records))
        in_pareto = [record.rewards if isinstance(record.rewards, list) else [record.rewards] for record in records]
        if not in_pareto:
            return None, None
        try:
            fitness = np.array(in_pareto)
            if fitness.shape[1] != 1 and nums is not None and len(in_pareto) > nums:
                # len must larger than nums, otherwise dead loop
                _, res, selected = SortAndSelectPopulation(fitness.T, nums)
            else:
                outs = pareto.eps_sort(fitness, maximize_all=True, attribution=True)
                res, selected = np.array(outs)[:, :-2], np.array(outs)[:, -1].astype(np.int32)
            return res.tolist(), selected.tolist()
        except Exception as ex:
            logging.error('No pareto_front_records found, ex=%s', ex)
            return [], []

    def get_step_records(self, step_name=None):
        """        Get step records.

        Retrieves records for a specific step or a list of steps. If no
        step_name is provided, it retrieves records for the default step name.
        It filters the records based on the provided step_name(s) and returns
        the filtered records.

        Args:
            step_name (str or list?): The name of the step or a list of step names to filter records.
                Defaults to None.

        Returns:
            list: A list of records filtered based on the provided step_name(s).
        """
        if not step_name:
            step_name = General.step_name
        records = self.all_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps, records))
        return records

    def get_pareto_front_records(self, step_name=None, nums=None, selected_key=None):
        """        Get Pareto Front Records.

        This function retrieves the Pareto front records based on the provided
        step name, numeric values, and selected key.

        Args:
            step_name (str or list): The name of the step or a list of step names. Defaults to
                General.step_name if not provided.
            nums (int): Numeric values for Pareto front calculation.
            selected_key (list): A list of selected keys for filtering records.

        Returns:
            list: A list of Pareto front records.
        """
        if not step_name:
            step_name = General.step_name
        records = self.all_records
        if selected_key is not None:
            new_records = []
            selected_key.sort()
            for record in records:
                record._objective_keys.sort()
                if record._objective_keys == selected_key:
                    new_records.append(record)
            records = new_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps and x.performance is not None, records))
        outs, selected = self.pareto_front(step_name, nums, records=records)
        if not outs:
            return []
        else:
            return [records[idx] for idx in selected]

    @classmethod
    def restore(cls):
        """        Transfer cvs_file to records.

        This function transfers the data from a CSV file to the records. It
        first checks if the file exists, then loads the data from the file and
        assigns it to the class attributes.

        Args:
            cls: The class instance.
        """
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".reports")
        if os.path.exists(_file):
            with open(_file, "rb") as f:
                data = fickling.load(f)
            cls._hist_records = data[0]
            cls.__instances__ = data[1]

    def backup_output_path(self):
        """        Back up output to a local path.

        This function backs up the output to a local path by copying the
        contents of the local output path to a backup path.
        """
        backup_path = TaskOps().backup_base_path
        if backup_path is None:
            return
        FileOps.copy_folder(TaskOps().local_output_path, backup_path)

    def output_pareto_front(self, step_name, desc=True, weights_file=True, performance=True):
        """        Save the Pareto front records.

        This method saves the Pareto front records based on the provided step
        name.

        Args:
            step_name (str): The name of the step for which Pareto front records are to be saved.
            desc (bool?): Flag to indicate whether to include description in the output. Defaults
                to True.
            weights_file (bool?): Flag to indicate whether to include weights file in the output. Defaults
                to True.
            performance (bool?): Flag to indicate whether to include performance metrics in the output.
                Defaults to True.
        """
        logging.debug("All records in report, records={}".format(self.all_records))
        records = deepcopy(self.get_pareto_front_records(step_name))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump pareto front records, report is emplty.")
            return
        self._output_records(step_name, records, desc, weights_file, performance)

    def output_step_all_records(self, step_name, desc=True, weights_file=True, performance=True):
        """        Output step all records.

        This method retrieves all records from the report and filters them based
        on the provided step_name. It then outputs the filtered records along
        with additional information if specified.

        Args:
            step_name (str): The name of the step to filter records for.
            desc (bool?): Whether to include description in the output. Defaults to True.
            weights_file (bool?): Whether to include weights file in the output. Defaults to True.
            performance (bool?): Whether to include performance metrics in the output. Defaults to True.
        """
        records = self.all_records
        logging.debug("All records in report, records={}".format(self.all_records))
        records = list(filter(lambda x: x.step_name == step_name, records))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump records, report is emplty.")
            return
        self._output_records(step_name, records, desc, weights_file, performance)
        logging.info(self.print_best(step_name))

    def _output_records(self, step_name, records, desc=True, weights_file=False, performance=False):
        """        Dump records.

        This function takes in the step name, records to be dumped, and optional
        flags for including description, weights file, and performance data. It
        serializes each record, selects specific columns, creates a DataFrame,
        saves the data to a CSV file, and copies additional files based on the
        flags.

        Args:
            step_name (str): The name of the step for organizing the output.
            records (list): List of records to be dumped.
            desc (bool?): Flag to include description files. Defaults to True.
            weights_file (bool?): Flag to include model weights files. Defaults to False.
            performance (bool?): Flag to include performance data files. Defaults to False.
        """
        columns = ["worker_id", "performance", "desc"]
        outputs = []
        for record in records:
            record = record.serialize()
            _record = {}
            for key in columns:
                _record[key] = record[key]
            outputs.append(deepcopy(_record))
        data = pd.DataFrame(outputs)
        step_path = FileOps.join_path(TaskOps().local_output_path, step_name)
        FileOps.make_dir(step_path)
        _file = FileOps.join_path(step_path, "output.csv")
        try:
            data.to_csv(_file, index=False)
        except Exception:
            logging.error("Failed to save output file, file={}".format(_file))
        for record in outputs:
            worker_id = record["worker_id"]
            worker_path = TaskOps().get_local_worker_path(step_name, worker_id)
            outputs_globs = []
            if desc:
                outputs_globs += glob.glob(FileOps.join_path(worker_path, "desc_*.json"))
            if weights_file:
                outputs_globs += glob.glob(FileOps.join_path(worker_path, "model_*"))
            if performance:
                outputs_globs += glob.glob(FileOps.join_path(worker_path, "performance_*.json"))
            for _file in outputs_globs:
                if os.path.isfile(_file):
                    FileOps.copy_file(_file, step_path)
                elif os.path.isdir(_file):
                    FileOps.copy_folder(_file, FileOps.join_path(step_path, os.path.basename(_file)))

    def dump(self):
        """        Dump report data to a file.

        This method prepares the report data and saves it to a CSV file. It also
        creates a backup of the data.

        Args:
            self: The instance of the ReportServer class.
        """
        try:
            _file = FileOps.join_path(TaskOps().step_path, "reports.csv")
            FileOps.make_base_dir(_file)
            data = self.all_records
            data_dict = {}
            for step in data:
                step_data = step.serialize().items()
                for k, v in step_data:
                    if k in data_dict:
                        data_dict[k].append(v)
                    else:
                        data_dict[k] = [v]

            data = pd.DataFrame(data_dict)
            data.to_csv(_file, index=False)
            _file = os.path.join(TaskOps().step_path, ".reports")
            _dump_data = [ReportServer._hist_records, ReportServer.__instances__]
            with open(_file, "wb") as f:
                pickle.dump(_dump_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.backup_output_path()
        except Exception:
            logging.warning(traceback.format_exc())

    def __repr__(self):
        """        Override the __repr__ function to return a string representation of the
        object.

        Returns:
            str: A string representation of the object.
        """
        return str(self.all_records)

    @classmethod
    def load_records_from_model_folder(cls, model_folder):
        """        Transfer JSON files to records from the specified model folder.

        Args:
            cls: The class instance.
            model_folder (str): The path to the model folder containing JSON files.

        Returns:
            list: A list of ReportRecord instances created from the JSON files.
        """
        if not model_folder or not os.path.exists(model_folder):
            logging.error("Failed to load records from model folder, folder={}".format(model_folder))
            return []
        records = []
        pattern = FileOps.join_path(model_folder, "desc_*.json")
        files = glob.glob(pattern)
        for _file in files:
            try:
                with open(_file) as f:
                    worker_id = _file.split(".")[-2].split("_")[-1]
                    weights_file = os.path.join(os.path.dirname(_file), "model_{}.pth".format(worker_id))
                    if os.path.exists(weights_file):
                        sample = dict(worker_id=worker_id, desc=json.load(f), weights_file=weights_file)
                    else:
                        sample = dict(worker_id=worker_id, desc=json.load(f))
                    record = ReportRecord().load_dict(sample)
                    records.append(record)
            except Exception as ex:
                logging.info('Can not read records from json because {}'.format(ex))
        return records

    @classmethod
    def _run_monitor_thread(cls):
        """Start a monitor thread to run the _monitor_thread function.

        This function starts a new thread to run the _monitor_thread function
        with the provided instances.

        Args:
            cls (class): The class instance to pass to the _monitor_thread function.

        Returns:
            Thread: The monitor thread that has been started.
        """

        logging.info("Start monitor thread.")
        monitor_thread = Thread(target=ReportServer._monitor_thread, args=(cls.__instances__,))
        monitor_thread.daemon = True
        monitor_thread.start()
        return monitor_thread

    @staticmethod
    def _monitor_thread(report_server):
        """Monitor thread function that continuously checks for updates in shared
        memory variables
        and adds new records to the report server.

        Args:
            report_server (ReportServer): An instance of the ReportServer class.
        """

        while report_server and report_server._thread_runing:
            watched_vars = deepcopy(ReportServer.__variables__)
            saved_records = report_server.all_records
            for var in watched_vars:
                step_name, worker_id = var.split(".")
                if step_name != General.step_name:
                    continue
                record_dict = None
                try:
                    record_dict = ShareMemory(var).get()
                except:
                    logging.warn("Failed to get record, step name: {}, worker id: {}.".format(step_name, worker_id))
                if record_dict:
                    record = ReportRecord().from_dict(record_dict)
                    code = record.code
                    saved_record = list(filter(
                        lambda x: x.step_name == step_name and str(x.worker_id) == str(worker_id), saved_records))
                    if not saved_record or record.code != saved_record[0].code:
                        report_server.add(record)
                        ReportServer().dump()
                    ShareMemory(var).close()
            time.sleep(0.2)
