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
import sys

import torch


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


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle. Copied from Pytorch3D.
    https://github.com/facebookresearch/pytorch3d/blob/f34104cf6ebefacd7b7e07955ee7aaa823e616ac/pytorch3d/transforms/rotation_conversions.py#L525

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
