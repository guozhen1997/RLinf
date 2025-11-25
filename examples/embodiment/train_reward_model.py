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

"""Training script for frame-based reward classifier model."""

import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from rlinf.data.datasets.reward import RewardDataset
from rlinf.models.reward_model.reward_classifier import BinaryRewardClassifier


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images_dict, labels in pbar:
        # Move to device
        images_dict = {k: v.to(device) for k, v in images_dict.items()}
        labels = labels.to(device).unsqueeze(1)  # [B] -> [B, 1]

        # Forward pass
        logits = model(images_dict, train=True)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": loss.item(),
                "acc": correct / total if total > 0 else 0.0,
            }
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def visualize_positive_samples(dataset, output_dir="positive_samples_vis"):
    """Visualize all samples with label=1 (positive samples).
    
    Args:
        dataset: RewardDataset instance
        output_dir: Directory to save visualization images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Visualizing positive samples (label=1)...")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")
    
    positive_samples = []
    for idx in range(len(dataset.samples)):
        traj_path, frame_idx, label = dataset.samples[idx]
        if label >= 0.5:  # Positive label
            positive_samples.append((traj_path, frame_idx, label))
    
    print(f"Found {len(positive_samples)} positive samples to visualize")
    
    if len(positive_samples) == 0:
        print("No positive samples found!")
        return
    
    # Visualize each positive sample
    for sample_idx, (traj_path, frame_idx, label) in enumerate(tqdm(positive_samples, desc="Saving images")):
        # Determine which directory this trajectory comes from
        traj_path_obj = Path(traj_path)
        if str(dataset.positive_dir) in str(traj_path_obj.parent):
            dir_name = "positive"
        elif str(dataset.negative_dir) in str(traj_path_obj.parent):
            dir_name = "negative"
        else:
            # Extract directory name from path
            dir_name = traj_path_obj.parent.name
        
        # Get trajectory filename (without extension)
        traj_filename = traj_path_obj.stem
        
        # Construct image filename: dir_name_traj_filename_frame_idx.png
        image_filename = f"{dir_name}_{traj_filename}_frame{frame_idx:03d}.png"
        image_path = output_path / image_filename
        
        # Load and save the image
        traj_data = np.load(traj_path, allow_pickle=True).item()
        frame_image = traj_data['images'][dataset.image_key][frame_idx]
        
        # Convert to numpy array if needed
        if isinstance(frame_image, torch.Tensor):
            frame_image = frame_image.cpu().numpy()
        
        # Ensure shape is [H, W, C] for PIL
        if len(frame_image.shape) == 3:
            if frame_image.shape[0] == dataset.image_size[0]:  # [C, H, W]
                frame_image = np.transpose(frame_image, (1, 2, 0))  # [H, W, C]
            # else already [H, W, C]
        
        # Normalize to [0, 255] if needed
        if frame_image.dtype == np.float32 or frame_image.dtype == np.float64:
            if frame_image.max() <= 1.0:
                frame_image = (frame_image * 255).astype(np.uint8)
            else:
                frame_image = np.clip(frame_image, 0, 255).astype(np.uint8)
        else:
            frame_image = np.clip(frame_image, 0, 255).astype(np.uint8)
        
        # Handle grayscale (single channel) or RGB
        if len(frame_image.shape) == 2:
            # Grayscale
            pass
        elif len(frame_image.shape) == 3:
            if frame_image.shape[2] == 1:
                frame_image = frame_image.squeeze(2)
            elif frame_image.shape[2] == 3:
                # RGB - ensure it's uint8
                pass
        
        # Save image
        Image.fromarray(frame_image).save(image_path)
    
    print(f"Saved {len(positive_samples)} positive sample images to {output_path}")
    print(f"{'='*60}\n")


@hydra.main(version_base="1.1", config_path="config", config_name="train_reward_model")
def main(cfg: DictConfig) -> None:
    """Train reward classifier model using Hydra configuration.
    
    Args:
        cfg: Configuration object from Hydra
    """
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    output_dir = os.path.dirname(cfg.output_checkpoint)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create model (currently only supports ResNet10, but can be extended)
    if cfg.backbone != "resnet10":
        print(f"Warning: Only resnet10 is currently supported. Using resnet10 instead of {cfg.backbone}")

    model = BinaryRewardClassifier(
        image_keys=[cfg.image_key],
        image_size=cfg.image_size,
        hidden_dim=cfg.hidden_dim,
        num_spatial_blocks=cfg.num_spatial_blocks,
        pretrained_encoder_path=cfg.get("pretrained_encoder_path", None),
        use_pretrain=cfg.get("pretrained_encoder_path", None) is not None,
        freeze_encoder=True,
    )

    # Auto-detect device if not specified
    device = cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create dataset
    train_dataset = RewardDataset(
        positive_dir=cfg.positive_dir,
        negative_dir=cfg.negative_dir,
        image_key=cfg.image_key,
        image_size=cfg.image_size,
        device=device,
    )
    
    # Visualize positive samples if requested
    if cfg.get("visualize_positive", False):
        vis_output_dir = cfg.get("vis_output_dir", "positive_samples_vis")
        visualize_positive_samples(train_dataset, vis_output_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True if device == "cuda" else False,
    )

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    best_train_acc = 0.0
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Save best model
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            torch.save(checkpoint, cfg.output_checkpoint)
            print(f"Saved best model with train_acc: {train_acc:.4f} to {cfg.output_checkpoint}")

    print("\nTraining completed!")
    print(f"Best model saved to {cfg.output_checkpoint}")


if __name__ == "__main__":
    main()


