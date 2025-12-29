import argparse
import os
import queue
import random
import socket
import subprocess
import threading
import time

import torch

from util.my import cur_time_str


def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--conda_env_name", type=str, default="SynLeaF", help="conda's environment name")
    parser.add_argument("--single_card_memory", type=int, default=48, help="unit: GB")
    parser.add_argument("--card_groups", nargs="+", type=str, default=["0,1", "2,3", "4,5", "6,7"], help="list of card groups. two in one group, separated by commas in each group")
    parser.add_argument("--worker_count_per_card_group", type=int, default=24)
    parser.add_argument("--base_port", type=int, default=33333, help="start port for Accelerate")

    parser.add_argument(
        "--task_types",
        nargs="+",
        type=str,
        default=["only_omics", "only_kg"],
        # default=["umt", "ume"],
        help="run `only_omics` and `only_kg` before `umt` and `ume`",
    )
    parser.add_argument(
        "--cancers",
        nargs="+",
        type=str,
        default=["BRCA", "CESC", "COAD", "KIRC", "LAML", "LUAD", "OV", "SKCM", "pan"],
        help="one or more cancer types",
    )
    parser.add_argument("--task_folder_prex", type=str, default="default", help="the prefix of task folder to distinguish different configurations")
    parser.add_argument("--addition_unified_params", type=str, default="", help="effective for all datasets, eg. --hid_dim 128")

    return parser.parse_args()


def generate_tasks(args):
    task_folder_prex = args.task_folder_prex
    addition_unified_params = args.addition_unified_params
    task_types = args.task_types
    cancers = args.cancers

    # Specific hyperparameters, equivalent to modifying on top of default parameters in train.py
    custom_hyper_parameters = {
        "BRCA": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "CESC": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "COAD": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "KIRC": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "LAML": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "LUAD": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "OV": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "SKCM": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "pan": {
            "cv_1": {"epochs": 30, "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
            "cv_2": {"epochs": 30, "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
            "cv_3": {"epochs": 30, "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
        },
    }

    # Concurrent running parameters (sum of multiple cards)
    mem_usage_dict = {
        "only_omics": {
            "single": 6,
            "pan": 14,
        },
        "only_kg": {
            "single": 20,
            "pan": 22,
        },
        "umt": {
            "single": 20,
            "pan": 35,
        },
        "ume": {
            "single": 8,
            "pan": 16,
        },
    }

    # Generate tasks
    total_tasks = []
    for task_index, task_type in enumerate(task_types):
        for cancer_type in cancers:
            for cv in range(1, 3 + 1):
                if cancer_type != "pan" and cv == 3:  # Single cancer does not run cv3
                    continue

                for fold in range(1, 5 + 1):
                    mem_usage = mem_usage_dict[task_type]["pan" if cancer_type == "pan" else "single"]
                    folder_name = f"{task_folder_prex}_{cancer_type}_cv_{cv}_fold_{fold}_{task_type}"

                    specific_param_dict = {}
                    if cancer_type in custom_hyper_parameters and f"cv_{cv}" in custom_hyper_parameters[cancer_type]:
                        specific_param_dict = custom_hyper_parameters[cancer_type][f"cv_{cv}"]
                    specific_cmd_args = " ".join(f"--{key} {value}" for key, value in specific_param_dict.items())
                    if addition_unified_params.strip() != "":
                        specific_cmd_args = f"{specific_cmd_args} {addition_unified_params}"

                    task_type_str = task_type
                    if task_type == "umt" or task_type == "ume":
                        task_type_str = f"{task_type} --omics_ckpt_path ../result/{task_folder_prex}_{cancer_type}_cv_{cv}_fold_{fold}_only_omics/checkpoint.pth --kg_ckpt_path ../result/{task_folder_prex}_{cancer_type}_cv_{cv}_fold_{fold}_only_kg/checkpoint.pth"

                    command = f"train.py --cancer_type {cancer_type} --metric {cv} --train_fold {fold} {specific_cmd_args} --task_type {task_type_str} --specify_result_saving_folder {folder_name} > ../result/{folder_name}/train.log 2>&1"

                    total_tasks.append(
                        {
                            "mem_usage": mem_usage,
                            "command": command,
                            "folder_name": folder_name,
                        }
                    )

    return total_tasks


def get_conda_env_vars(conda_env_name):
    """
    Get all environment variables of specified Conda environment
    """

    command = f"conda run -n {conda_env_name} env"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    return env_vars


class PortManager:
    def __init__(self, base_port):
        self.port = base_port
        self.lock = threading.Lock()

    def _is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def get_free_port(self):  # Port always increases, so no reuse
        with self.lock:
            while self._is_port_in_use(self.port):
                self.port += 1
            free_port = self.port
            self.port += 1
            return free_port


class Worker(threading.Thread):
    def __init__(self, worker_id, card_group, port_manager, env_vars, task_queue, wait_time_min=8.0, wait_time_max=16.0):
        """
        :param worker_id: Unique identifier for worker
        :param card_group: Belonging card group
        :param port_manager: Port manager
        :param env_vars: Current environment variables
        :param task_queue: Current task queue
        :param wait_time_min: Minimum wait time for failure retry (seconds)
        :param wait_time_max: Maximum wait time for failure retry (seconds)
        """
        super().__init__()
        self.daemon = True  # Used by parent class. When main thread exits, daemon threads automatically terminate regardless of task completion.

        self.worker_id = worker_id
        self.card_group = card_group
        self.port_manager = port_manager
        self.port = self.port_manager.get_free_port()
        self.env_vars = env_vars
        self.task_queue = task_queue
        self.wait_time_min = wait_time_min
        self.wait_time_max = wait_time_max

    def run(self):
        """
        Method inherited from Thread
        """

        while True:
            try:
                task = self.task_queue.get(block=False)  # non-blocking
            except queue.Empty:
                break  # If queue is empty, end thread
            self.work(task)  # Execute command
            self.task_queue.task_done()
        print(f"[{cur_time_str()}][Worker-{self.worker_id}] running completed.")

    def work(self, task):
        """
        Execute single task
        """

        mem_usage = task["mem_usage"]  # Memory usage
        command = task["command"]  # Execution command
        folder_name = task["folder_name"]  # Result folder name

        while True:  # Loop to allocate memory
            if self.card_group.allocate_memory(mem_usage):
                self.execute_command(folder_name, command)
                self.card_group.release_memory(mem_usage)
                return
            else:
                time.sleep(random.uniform(self.wait_time_min, self.wait_time_max))

    def execute_command(self, folder_name, command):
        max_retries = 5
        for attempt in range(max_retries):
            gpu_ids = self.card_group.get_gpu_ids()
            gpu_count = len(gpu_ids)
            gpu_ids_str = ",".join(map(str, gpu_ids))
            commmand_prex = f"accelerate launch --main_process_port {self.port} --num_processes {gpu_count} --gpu_ids {gpu_ids_str}"
            full_command = f"{commmand_prex} {command}"

            try:
                print(f"{self.log_prex()} prepare to execute (attempt {attempt + 1}/{max_retries}): {full_command}")
                if attempt == 0:
                    subprocess.run(f"mkdir -p ../result/{folder_name}", shell=True, check=True, env=self.env_vars)

                subprocess.run(full_command, shell=True, check=True, env=self.env_vars)

                print(f"{self.log_prex()} finished to execute: {full_command}")
                return

            except subprocess.CalledProcessError as e:
                print(f"{self.log_prex()}[err_code: {e.returncode}] command execution FAILED: {full_command}")
                if attempt < max_retries - 1:
                    old_port = self.port
                    self.port = self.port_manager.get_free_port()
                    print(f"[{cur_time_str()}][Worker-{self.worker_id}][Port-{old_port}] Retrying with new port {self.port}...")
                    time.sleep(random.uniform(1, 3))
                else:
                    print(f"[{cur_time_str()}][Worker-{self.worker_id}] All {max_retries} retries failed. Giving up on task.")

    def log_prex(self):
        return f"[{cur_time_str()}][Worker-{self.worker_id}][Port-{self.port}]"


class CardGroup:
    def __init__(self, single_card_memory, gpu_ids):
        """
        Initialize CardGroup class, card group consists of multiple cards, e.g., two cards
        :param single_card_memory: GPU single card memory capacity, unit: GB, default same for each card
        :param gpu_ids: Device IDs used, e.g., [0,1]
        """

        self.max_memory = single_card_memory * len(gpu_ids)
        self.used_memory = 0  # Currently used memory
        self.lock = threading.Lock()

        self.gpu_ids = gpu_ids

    def allocate_memory(self, memory):
        with self.lock:
            if self.used_memory + memory <= self.max_memory:
                self.used_memory += memory
                return True
            else:
                return False

    def release_memory(self, memory):
        with self.lock:
            if memory <= self.used_memory:
                self.used_memory -= memory

    def get_used_memory(self):
        return self.used_memory

    def get_available_memory(self):
        return self.max_memory - self.used_memory

    def get_gpu_ids(self):
        return self.gpu_ids


def main():
    args = init_argparse()
    print("parameters:", args)

    # Get Conda environment variables (execute only once)
    print("loading conda environment...")
    env_vars = os.environ.copy()
    env_vars.update(get_conda_env_vars(args.conda_env_name))  # Append Conda variables

    # Generate tasks
    task_queue = queue.Queue()
    print("generating tasks...")
    tasks = generate_tasks(args)
    print(f"generated done. total tasks: {len(tasks)}")
    torch.save(tasks, f"../result/task_list_{args.task_folder_prex}.pt")  # Used for manual handling after disconnection
    print(f"success to save tasks list data to:", f"../result/task_list_{args.task_folder_prex}.pt")

    # Put all tasks into queue
    for t in tasks:
        task_queue.put(t)

    port_manager = PortManager(args.base_port)
    total_workers = []  # Total workers, 2D array
    worker_id_counter = 0
    for gpu_ids_str in args.card_groups:
        cur_workers = []
        gpu_ids = list(map(int, gpu_ids_str.split(",")))
        card_group = CardGroup(single_card_memory=args.single_card_memory, gpu_ids=gpu_ids)

        for i in range(args.worker_count_per_card_group):
            worker_id_counter += 1
            cur_workers.append(Worker(worker_id_counter, card_group, port_manager, env_vars, task_queue))

        total_workers.append(cur_workers)

    workers = []  # Worker sequence prepared to start
    for i in range(args.worker_count_per_card_group):
        for j in range(len(args.card_groups)):
            workers.append(total_workers[j][i])

    # Start Worker threads
    for worker in workers:
        worker.start()
        time.sleep(1.5)  # Prevent duplicate random folder names

    task_queue.join()
    for worker in workers:
        worker.join()

    print("all tasks running completed, program exit!")


if __name__ == "__main__":
    main()
