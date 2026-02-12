# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import os
import subprocess
import sys

import torch


def get_gpu_numa_node(gpu_id: int) -> int:
    try:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            # Get PCI bus info
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId.decode("utf-8")
        except ImportError:
            # Fallback to nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            pci_bus_id = result.stdout.strip()

        # Extract bus number from PCI bus ID (format: 0000:XX:YY.Z)
        bus_number = pci_bus_id.split(":")[1]

        # Get NUMA node from sysfs
        numa_node_path = f"/sys/bus/pci/devices/0000:{bus_number}:00.0/numa_node"
        if os.path.exists(numa_node_path):
            with open(numa_node_path, "r") as f:
                numa_node = int(f.read().strip())
                if numa_node >= 0:
                    return numa_node

        # Fallback: try to get from lscpu
        result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
        numa_nodes = 0
        for line in result.stdout.split("\n"):
            if "NUMA node(s):" in line:
                numa_nodes = int(line.split(":")[1].strip())
                break

        # If we can't determine the exact NUMA node, distribute evenly
        return gpu_id % numa_nodes if numa_nodes > 0 else 0

    except Exception as e:
        print(f"Warning: Could not determine NUMA node for GPU {gpu_id}: {e}")
        return 0


def get_numa_cpus(numa_node: int) -> list:
    try:
        # Read from sysfs
        cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        if os.path.exists(cpulist_path):
            with open(cpulist_path, "r") as f:
                cpulist = f.read().strip()

            # Parse CPU list (e.g., "0-7,16-23" or "0,1,2,3")
            cpus = []
            for part in cpulist.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    cpus.extend(range(start, end + 1))
                else:
                    cpus.append(int(part))
            return cpus
    except Exception as e:
        print(f"Warning: Could not get CPU list for NUMA node {numa_node}: {e}")

    # Fallback: return all available CPUs
    return list(range(os.cpu_count() or 1))


def set_process_numa_affinity(gpu_id: int) -> None:
    try:
        numa_node = get_gpu_numa_node(gpu_id)
        cpus = get_numa_cpus(numa_node)

        if not cpus:
            print(f"Warning: No CPUs found for NUMA node {numa_node}")
            return

        os.sched_setaffinity(0, cpus)
        try:
            subprocess.run(
                ["numactl", "--membind", str(numa_node), "--"],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass  # numactl not available, that's ok

    except Exception as e:
        print(f"Warning: Could not set NUMA affinity for GPU {gpu_id}: {e}")


def recursive_to_own(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_shared() else obj
    elif isinstance(obj, list):
        return [recursive_to_own(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_own(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_own(v) for k, v in obj.items()}
    else:
        return obj


def force_gc_tensor(tensor):
    if not torch.is_tensor(tensor):
        return

    try:
        ref_count = sys.getrefcount(tensor)
        for _ in range(ref_count + 10):
            ctypes.pythonapi.Py_DecRef(ctypes.py_object(tensor))

    except Exception as e:
        print(f"Error during force delete: {e}")


def cleanup_cuda_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            force_gc_tensor(obj)
    gc.collect()
    torch.cuda.empty_cache()


def get_batch_rng_state(batched_rng):
    state = {
        "rngs": batched_rng.rngs,
    }
    return state


def set_batch_rng_state(state: dict):
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG

    return BatchedRNG.from_rngs(state["rngs"])
