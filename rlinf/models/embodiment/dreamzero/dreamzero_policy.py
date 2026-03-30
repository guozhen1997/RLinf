import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from hydra.utils import instantiate
from tqdm import tqdm
import numpy as np
import torch
from tianshou.data import Batch
from omegaconf import OmegaConf, DictConfig
import os
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
# Ensure groot is importable (dreamzero repo structure)
def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    dreamzero_root = dreamzero_root / "DreamZero"
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


class DreamZeroPolicy(BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead.

    - predict_action_batch: for eval (inference)
    - forward: for SFT training (returns loss)
    """

    def __init__(
        self,
        model_path: str,
        device: str | int = "cuda",
        eval_bf16: bool = True,
        force_identity_backbone: bool = True,
        tokenizer_path: str = "google/umt5-xxl",
        max_seq_len: int = 512,
    ):
        _ensure_groot_importable()

        from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
        from groot.vla.data.schema import DatasetMetadata
        from groot.vla.data.transform import ComposedModalityTransform

        self.model_path = Path(model_path)
        self.device = torch.device(device if isinstance(device, str) else f"cuda:{device}")
        self.eval_bf16 = eval_bf16
        self._tokenizer = None
        exp_cfg_dir = self.model_path / "experiment_cfg"
        train_cfg_path = exp_cfg_dir / "conf.yaml"
        train_cfg = OmegaConf.load(train_cfg_path)
        self.train_cfg = train_cfg
        self.eval_bf16 = self.train_cfg.get("eval_bf16", False)

        # Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Force IdentityBackbone for lightweight structure
        if force_identity_backbone:
            config_dict["backbone_cfg"] = {
                "_target_": "groot.vla.model.dreamzero.backbone.identity.IdentityBackbone"
            }

        config = VLAConfig(**config_dict)

        # Disable defer_lora_injection for immediate loading
        if "config" in config.action_head_cfg and isinstance(config.action_head_cfg["config"], dict):
            config.action_head_cfg["config"]["defer_lora_injection"] = False
            config.action_head_cfg["config"]["skip_component_loading"] = True

        # Load model: use custom loading when forcing IdentityBackbone (from_pretrained ignores config)
        if force_identity_backbone:
            self.model = self._load_model_with_config(str(self.model_path), config)
        else:
            self.model = VLA.from_pretrained(str(self.model_path))

        if eval_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)
        self.model = self.model.to(device=self.device)

        if hasattr(self.model, "post_initialize"):
            try:
                self.model.post_initialize()
            except Exception as e:
                print(f"post_initialize skipped: {e}")

        self.action_horizon = self.model.action_horizon
        self.action_dim = self.model.action_dim


        # 2. Load the action, video, and state transforms
        # 2.1. Load the metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)
        embodiment_tag = next(iter(metadatas.keys()))
        metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

        # 2.2. Get the eval transforms

        train_cfg.transforms[embodiment_tag].transforms[-1].tokenizer_path = tokenizer_path
        eval_transform = instantiate(train_cfg.transforms[embodiment_tag])
        assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)
        eval_transform.eval()
        self.eval_transform = eval_transform

    def eval(self):
        self.model.eval()
        return self
    
    def cuda(self):
        self.model = self.model.cuda()
        return self

    def to(self, device: str | int):
        self.model = self.model.to(device)
        return self 

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs
        normalized_input = self.eval_transform(obs)
        batch.normalized_obs = normalized_input
        return batch
    
    def unapply(self, batch: Batch, obs: dict = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.eval_transform.unapply(
            dict(action=batch.normalized_action.cpu())
        )
        
        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.train_cfg.get('relative_action', False)
        relative_action_per_horizon = self.train_cfg.get('relative_action_per_horizon', False)
        relative_action_keys = self.train_cfg.get('relative_action_keys', [])
        if (relative_action or relative_action_per_horizon) and relative_action_keys and obs is not None:
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"
                
                if action_key not in unnormalized_action:
                    continue
                
                # Try to find the state data - check multiple possible key formats
                last_state = None
                
                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if 'state' in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break
                    
                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and 'state' in obs:
                        state_data = obs['state']
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None
                        
                        if state_dim == action_dim:
                            last_state = state_data
                
                if last_state is None:
                    continue
                    
                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()
                
                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep
                
                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(last_state, axis=-2)  # Add horizon dimension
                
                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = unnormalized_action[action_key] + last_state
        
        batch.act = unnormalized_action
        return batch

    def _load_model_with_config(self, model_path: str, config) -> "VLA":
        """Load VLA with custom config (e.g. IdentityBackbone) and weights."""
        from safetensors.torch import load_file

        from groot.vla.model.dreamzero.base_vla import VLA

        state_dict = {}
        safetensors_path = Path(model_path) / "model.safetensors"
        safetensors_index_path = Path(model_path) / "model.safetensors.index.json"
        if safetensors_index_path.exists():
            with open(safetensors_index_path) as f:
                index = json.load(f)
            for shard_file in set(index["weight_map"].values()):
                shard_path = Path(model_path) / shard_file
                state_dict.update(load_file(str(shard_path)))
        elif safetensors_path.exists():
            state_dict.update(load_file(str(safetensors_path)))
        else:
            raise FileNotFoundError(f"No weights at {model_path}")

        model = VLA(config)
        if hasattr(model, "post_initialize"):
            model.post_initialize()

        has_base_layer = any(".base_layer." in k for k in state_dict)
        if has_base_layer:
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        return model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer
            self._tokenizer = HuggingfaceTokenizer(
                name=self.tokenizer_path,
                seq_len=self.max_seq_len,
                clean="whitespace",
            )
        return self._tokenizer
    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do bf16 cast if needed
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)
        return normalized_input
    
    def _droid_observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to model input"""
        # RLinf obs -> DreamZero obs
        main = env_obs["main_images"]
        extra = env_obs.get("extra_view_images", None)
        wrist = env_obs.get("wrist_images",None)
        states = env_obs.get("states", None)
        task_desc = env_obs.get("task_descriptions", None)
        def _ensure_video_bt_hwc(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.ndim == 4:
                x = x[:, None, ...]   # [B,1,H,W,C]
            elif x.ndim != 5: # expect [B,T,H,W,C]
                raise ValueError(f"Unexpected image shape: {x.shape}")
            return x

        import cv2

        def _resize_bt_hwc_uint8(x, h=180, w=320):
            B = x.shape[0]
            out = np.empty((B, h, w, 3), dtype=np.uint8)
            for b in range(B):
                frame = x[b]
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out[b] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return out


        if torch.is_tensor(main):
            main_np = main.detach().cpu().numpy()
        else:
            main_np = np.asarray(main)
        main_np = _resize_bt_hwc_uint8(main_np)

        B = main_np.shape[0]

        # states -> joint/gripper (according to panda-droid 8D convention)
        if states is not None:
            if torch.is_tensor(states):
                s_np = states.detach().cpu().numpy()
            else:
                s_np = np.asarray(states)
        else:
            s_np = np.zeros((B, 8), dtype=np.float32)

        if s_np.ndim == 1:
            s_np = s_np.reshape(1, -1)

        # prioritize joint(7)+gripper(1) if possible, otherwise fallback to 7D joint
        if s_np.shape[-1] >= 8:
            joint = s_np[:, :7]
            gripper = s_np[:, 7:8]
        elif s_np.shape[-1] >= 7:
            joint = s_np[:, :7]
            gripper = np.zeros((B, 1), dtype=s_np.dtype)
        else:
            joint = np.zeros((B, 7), dtype=np.float32)
            gripper = np.zeros((B, 1), dtype=np.float32)

        prompts = task_desc if task_desc is not None else [""] * B
        if isinstance(prompts, str):
            prompts = [prompts] * B

        converted_obs = {
            "video.exterior_image_1_left": main_np,                     # [B,H,W,C]
            "video.exterior_image_2_left": main_np,                     # [B,H,W,C]
            "video.wrist_image_left": main_np,                         # [B,H,W,C]
            "state.joint_position": joint.astype(np.float64),        # [B,7]
            "state.gripper_position": gripper.astype(np.float64),    # [B,1]
            "annotation.language.action_text": list(prompts),        # list[str], len=B
        }
        return converted_obs
    
    def _libero_observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to model input for end-effector control"""
        main = env_obs["main_images"]
        wrist = env_obs.get("wrist_images", None)
        extra_view = env_obs.get("extra_view_images", None)
        states = env_obs.get("states", None)
        prompts = env_obs.get("task_descriptions", None)
        if torch.is_tensor(main):
            main = main.detach().cpu().numpy()
        else:
            main = np.asarray(main)
        B = main.shape[0]
        if wrist is not None:
            if torch.is_tensor(wrist):
                wrist = wrist.detach().cpu().numpy()
            else:
                wrist = np.asarray(wrist)
        import cv2
        def _resize_bt_hwc_uint8(x, h=256, w=256):
            # x: [B,H,W,C
            B = x.shape[0]
            out = np.empty((B, h, w, 3), dtype=np.uint8)
            for b in range(B):
                frame = x[b]
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out[b] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return out
        main = _resize_bt_hwc_uint8(main)
        if wrist is not None:
            wrist = _resize_bt_hwc_uint8(wrist)
        if main.ndim == 4:
            main = main[:, None, ...]
        if wrist is not None and wrist.ndim == 4:
            wrist = wrist[:, None, ...]
        if states is not None:
            if torch.is_tensor(states):
                s_np = states.detach().cpu().numpy()
            else:
                s_np = np.asarray(states)
        else:
            s_np = np.zeros((B, 8), dtype=np.float32)
        if s_np.ndim == 1:
            s_np = s_np[None, :]
        elif s_np.ndim > 2:
            s_np = s_np.reshape(B, -1)
        s_np = s_np.astype(np.float32)
        state_bt = s_np[:, None, :]
        prompts = prompts if prompts is not None else [""] * B
        if isinstance(prompts, str):
            prompts = [prompts] * B
        converted_obs = {
            "video.image": main,                     # [B,H,W,C]
            "video.wrist_image": wrist,                     # [B,H,W,C]
            "state.state": state_bt,        # [B,1,8]
            "annotation.language.action_text": list(prompts),        # list[str], len=B
        }
        return converted_obs
    
    def _convert_action_to_droid(self, action_dict: dict) -> np.ndarray:
        """Convert DreamZero action dict to DROID action array.
        
        DreamZero format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        
        return action

    def predict_action_batch(self, env_obs, mode,**kwargs) -> np.ndarray:
        """
        input:
        env_obs:
            - main_images: [B,H,W,C] uint8
            - extra_view_images: [B,H,W,C]
            - states: [B,D]
            - task_descriptions: list[str] or None
        output:
        actions: np.ndarray [B, num_action_chunks, 8]  # 6ee + 1 gripper
        result: dict  # compatible with rollout interface"""

        B = env_obs["main_images"].shape[0]
        converted_obs = self._libero_observation_convert(env_obs)
        batch = Batch(obs=converted_obs)        
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.model.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        # Unnormalize actions (pass obs for relative action normalization)
        #batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs_for_relative)
        unnormalized_action = self.eval_transform.unapply(
            dict(action=normalized_action.cpu())
        )
        batch.act = unnormalized_action

        actions = batch.act["action.actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(actions.dtype)

        flat = torch.as_tensor(actions, dtype=torch.float32).reshape(actions.shape[0], -1).cpu()
        forward_inputs = {"action": flat}
        result = {
        "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
        "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
        "forward_inputs": forward_inputs,
        }
        return actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:

        """Default forward pass."""
        raise NotImplementedError

def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint.
    """
    model_path = cfg.actor.model.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DreamZero model_path does not exist: {model_path}. "
            "Please provide a valid checkpoint directory."
        )

    tokenizer_path = cfg.actor.model.get("tokenizer_path", "google/umt5-xxl")
    print("tokenizer_path", tokenizer_path)
    max_seq_len = cfg.actor.model.get("max_seq_len", 512)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = DreamZeroPolicy(
        model_path=model_path,
        device=device,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )

    return model
