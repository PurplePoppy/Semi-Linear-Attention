from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except (AttributeError, RuntimeError):
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except (AttributeError, RuntimeError):
            pass

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:

            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            pass
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

class ResizeWithPad:

    def __init__(
        self,
        size: int,
        fill: int = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.size = int(size)
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        image = TF.resize(
            image,
            [new_height, new_width],
            interpolation=self.interpolation,
            antialias=True,
        )

        pad_left = (self.size - new_width) // 2
        pad_right = self.size - new_width - pad_left
        pad_top = (self.size - new_height) // 2
        pad_bottom = self.size - new_height - pad_top

        return TF.pad(
            image,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
            padding_mode="constant",
        )


def build_stat_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            ResizeWithPad(image_size),
            transforms.ToTensor(),
        ]
    )


def build_train_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    use_hflip: bool,
    rotation_degrees: float,
) -> transforms.Compose:
    aug: List = [
        transforms.Grayscale(num_output_channels=1),
        ResizeWithPad(image_size),
    ]

    if use_hflip:
        aug.append(transforms.RandomHorizontalFlip(p=0.5))
    if rotation_degrees > 0:
        aug.append(
            transforms.RandomRotation(
                degrees=rotation_degrees,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(aug)


def build_eval_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            ResizeWithPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def compute_mean_std(
    train_path: Path,
    train_indices: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> Tuple[List[float], List[float]]:

    stat_dataset = datasets.ImageFolder(
        str(train_path), transform=build_stat_transform(image_size)
    )
    stat_subset = Subset(stat_dataset, list(train_indices))

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        stat_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )

    channel_sum = torch.zeros(1, dtype=torch.float64)
    channel_sq_sum = torch.zeros(1, dtype=torch.float64)
    pixel_count = 0

    for images, _ in tqdm(loader, desc="Computing train mean/std", leave=False):
        images = images.to(dtype=torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images**2).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    if pixel_count == 0:
        raise RuntimeError("No pixels were found while computing normalization statistics.")

    mean = channel_sum / pixel_count
    variance = channel_sq_sum / pixel_count - mean**2
    std = torch.sqrt(torch.clamp(variance, min=1e-12))

    return [float(mean.item())], [float(std.item())]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class LaplacianEnhancement(nn.Module):

    def __init__(
        self,
        channels: int,
        alpha_init: float = 0.2,
        learnable_alpha: bool = True,
        alpha_max: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0 <= alpha_init < alpha_max:
            raise ValueError(
                f"alpha_init must satisfy 0 <= alpha_init < alpha_max; "
                f"got {alpha_init} and {alpha_max}."
            )

        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.channels = channels
        self.learnable_alpha = learnable_alpha
        self.alpha_max = float(alpha_max)

        if learnable_alpha:
            ratio = min(max(alpha_init / alpha_max, 1e-6), 1 - 1e-6)
            raw = math.log(ratio / (1.0 - ratio))
            self.alpha_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("alpha_fixed", torch.tensor(alpha_init, dtype=torch.float32))

    def current_alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return self.alpha_max * torch.sigmoid(self.alpha_raw)
        return self.alpha_fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        lap = F.conv2d(
            x_pad,
            self.weight,
            stride=1,
            padding=0,
            groups=self.channels,
        )
        return x + self.current_alpha() * lap


class SemiLinearAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        score_hidden_dim: int,
        gamma_init: float,
        gamma_max: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if abs(gamma_init) >= gamma_max:
            raise ValueError("abs(gamma_init) must be smaller than gamma_max.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.gamma_max = float(gamma_max)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(1, score_hidden_dim),
            nn.GELU(),
            nn.Linear(score_hidden_dim, 1),
        )

        init_ratio = gamma_init / gamma_max
        raw_init = 0.5 * math.log((1 + init_ratio) / (1 - init_ratio))
        self.gamma_raw = nn.Parameter(
            torch.full((num_heads, 1, 1), raw_init, dtype=torch.float32)
        )

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def current_gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.tanh(self.gamma_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        nonlinear_scores = self.score_mlp(scores.unsqueeze(-1)).squeeze(-1)
        semi_scores = scores + self.current_gamma().unsqueeze(0) * nonlinear_scores

        attention = torch.softmax(semi_scores, dim=-1)
        attention = self.attn_drop(attention)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemiLinearTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        score_hidden_dim: int,
        gamma_init: float,
        gamma_max: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SemiLinearAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            score_hidden_dim=score_hidden_dim,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ResNetLaplacianSemiLinearTransformer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        embed_dim: int = 128,
        transformer_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        score_hidden_dim: int = 16,
        gamma_init: float = 0.01,
        gamma_max: float = 1.0,
        lap_alpha: float = 0.2,
        learnable_lap_alpha: bool = True,
        lap_alpha_max: float = 0.5,
        classifier_dropout: float = 0.2,
        pos_base_size: int = 7,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self._current_channels = 64
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.token_projection = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.laplacian = LaplacianEnhancement(
            channels=embed_dim,
            alpha_init=lap_alpha,
            learnable_alpha=learnable_lap_alpha,
            alpha_max=lap_alpha_max,
        )

        self.pos_embed_2d = nn.Parameter(
            torch.zeros(1, embed_dim, pos_base_size, pos_base_size)
        )

        self.transformer = nn.Sequential(
            *[
                SemiLinearTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    score_hidden_dim=score_hidden_dim,
                    gamma_init=gamma_init,
                    gamma_max=gamma_max,
                )
                for _ in range(transformer_depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_module)
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [
            BasicBlock(self._current_channels, out_channels, stride=stride)
        ]
        self._current_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self._current_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.token_projection(x)
        x = self.laplacian(x)

        _, _, height, width = x.shape
        pos_embed = F.interpolate(
            self.pos_embed_2d,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )

        x = x + pos_embed
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        features = self.classifier_dropout(features)
        return self.classifier(features)

    def get_interpretable_parameters(self) -> Dict[str, object]:
        gamma_values = []
        for block in self.transformer:
            gamma_values.append(
                block.attention.current_gamma().detach().cpu().view(-1).numpy()
            )
        return {
            "laplacian_alpha": float(
                self.laplacian.current_alpha().detach().cpu().item()
            ),
            "semi_linear_gamma_per_block_and_head": np.stack(gamma_values).tolist(),
        }


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = -((1.0 - target_probs) ** self.gamma) * target_log_probs
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def compute_class_counts(targets: Sequence[int], indices: Sequence[int], num_classes: int):
    selected = np.asarray(targets, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(selected, minlength=num_classes)
    return counts


def compute_class_weights(counts: np.ndarray) -> torch.Tensor:
    if np.any(counts <= 0):
        raise ValueError(f"Every class must appear in the internal training split: {counts}")
    total = counts.sum()
    weights = total / (len(counts) * counts.astype(np.float64))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(
    loss_type: str,
    class_weights: torch.Tensor,
    label_smoothing: float,
    focal_gamma: float,
    device: torch.device,
) -> nn.Module:
    class_weights = class_weights.to(device)
    if loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    raise ValueError(f"Unsupported loss type: {loss_type}")

@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> EpochResult:
    return EpochResult(
        loss=float(loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        macro_recall=float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    loss: float,
) -> Tuple[Dict[str, float], np.ndarray, List[Dict[str, float]]]:
    num_classes = len(class_names)
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    overall = asdict(basic_metrics(y_true, y_pred, loss))
    overall.update(
        {
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        }
    )

    y_bin = label_binarize(y_true, classes=labels)
    try:
        overall["macro_roc_auc_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
        overall["weighted_roc_auc_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        )
    except ValueError:
        overall["macro_roc_auc_ovr"] = float("nan")
        overall["weighted_roc_auc_ovr"] = float("nan")

    try:
        overall["macro_pr_auc"] = float(
            average_precision_score(y_bin, y_prob, average="macro")
        )
    except ValueError:
        overall["macro_pr_auc"] = float("nan")

    per_class: List[Dict[str, float]] = []
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    specificities = []
    for class_idx, class_name in enumerate(class_names):
        tp = cm[class_idx, class_idx]
        fn = cm[class_idx, :].sum() - tp
        fp = cm[:, class_idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        specificities.append(specificity)

        row = {
            "class_index": class_idx,
            "class_name": class_name,
            "support": int(supports[class_idx]),
            "precision": float(precisions[class_idx]),
            "sensitivity_recall": float(recalls[class_idx]),
            "specificity": float(specificity),
            "f1": float(f1s[class_idx]),
        }

        if y_bin.ndim == 2 and y_bin.shape[1] == num_classes:
            try:
                row["roc_auc_ovr"] = float(
                    roc_auc_score(y_bin[:, class_idx], y_prob[:, class_idx])
                )
            except ValueError:
                row["roc_auc_ovr"] = float("nan")
            try:
                row["pr_auc"] = float(
                    average_precision_score(y_bin[:, class_idx], y_prob[:, class_idx])
                )
            except ValueError:
                row["pr_auc"] = float("nan")

        per_class.append(row)

    overall["macro_specificity"] = float(np.nanmean(specificities))
    return overall, cm, per_class


def stratified_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    if iterations <= 0:
        return {}

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == c) for c in range(num_classes)]
    if any(len(idx) == 0 for idx in class_indices):
        raise ValueError("All classes must occur in the test set for stratified bootstrap.")

    values: Dict[str, List[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "balanced_accuracy": [],
        "macro_roc_auc_ovr": [],
        "macro_pr_auc": [],
    }

    classes = np.arange(num_classes)
    for _ in tqdm(range(iterations), desc="Bootstrap 95% CI", leave=False):
        sampled_parts = [rng.choice(idx, size=len(idx), replace=True) for idx in class_indices]
        sampled = np.concatenate(sampled_parts)
        rng.shuffle(sampled)

        yt = y_true[sampled]
        yp = y_pred[sampled]
        ypr = y_prob[sampled]

        values["accuracy"].append(accuracy_score(yt, yp))
        values["macro_f1"].append(
            f1_score(yt, yp, average="macro", zero_division=0)
        )
        values["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        values["macro_roc_auc_ovr"].append(
            roc_auc_score(yt, ypr, multi_class="ovr", average="macro")
        )
        yt_bin = label_binarize(yt, classes=classes)
        values["macro_pr_auc"].append(
            average_precision_score(yt_bin, ypr, average="macro")
        )

    alpha = (1.0 - confidence) / 2.0
    output: Dict[str, Dict[str, float]] = {}
    for metric_name, metric_values in values.items():
        arr = np.asarray(metric_values, dtype=np.float64)
        output[metric_name] = {
            "lower": float(np.quantile(arr, alpha)),
            "upper": float(np.quantile(arr, 1.0 - alpha)),
            "confidence": confidence,
            "iterations": iterations,
        }
    return output


def tta_views(images: torch.Tensor, mode: str) -> List[torch.Tensor]:
    views = [images]
    if mode == "none":
        return views
    if mode == "hflip":
        views.append(torch.flip(images, dims=[3]))
        return views
    if mode == "rot3":
        views.append(TF.rotate(images, angle=3.0, interpolation=InterpolationMode.BILINEAR))
        views.append(TF.rotate(images, angle=-3.0, interpolation=InterpolationMode.BILINEAR))
        return views
    if mode == "hflip_rot3":
        views.append(torch.flip(images, dims=[3]))
        views.append(TF.rotate(images, angle=3.0, interpolation=InterpolationMode.BILINEAR))
        views.append(TF.rotate(images, angle=-3.0, interpolation=InterpolationMode.BILINEAR))
        return views
    raise ValueError(f"Unsupported TTA mode: {mode}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    tta_mode: str = "none",
    amp_enabled: bool = True,
    description: str = "Evaluate",
) -> Tuple[EpochResult, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []

    for images, labels in tqdm(dataloader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            if tta_mode == "none":
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
                loss = criterion(logits, labels)
            else:
                probability_list = []
                for view in tta_views(images, tta_mode):
                    logits_view = model(view)
                    probability_list.append(torch.softmax(logits_view, dim=1))
                probabilities = torch.stack(probability_list, dim=0).mean(dim=0)

                loss = F.nll_loss(
                    torch.log(probabilities.clamp_min(1e-8)),
                    labels,
                )

        predictions = probabilities.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)

    if y_prob.shape[1] != num_classes:
        raise RuntimeError(
            f"Probability output has {y_prob.shape[1]} columns, expected {num_classes}."
        )

    result = basic_metrics(y_true, y_pred, total_loss / max(total_samples, 1))
    return result, y_true, y_pred, y_prob


class WarmupCosineScheduler:

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        min_lr_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(1, total_epochs)
        self.warmup_epochs = max(0, warmup_epochs)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def step(self) -> None:
        self.last_epoch += 1
        epoch = self.last_epoch
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
            progress = min(
                1.0,
                max(0.0, (epoch - self.warmup_epochs + 1) / cosine_epochs),
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_samples = 0
    labels_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        labels_all.append(labels.detach().cpu().numpy())
        preds_all.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    return basic_metrics(y_true, y_pred, total_loss / max(total_samples, 1))


def metric_value(result: EpochResult, metric_name: str) -> float:
    if not hasattr(result, metric_name):
        raise ValueError(f"EpochResult has no metric named '{metric_name}'.")
    return float(getattr(result, metric_name))

class HistoryWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "train_balanced_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_balanced_accuracy",
            "elapsed_seconds",
        ]
        with self.path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def append(self, row: Dict[str, float]) -> None:
        with self.path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_per_class_metrics(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    sample_paths: Optional[Sequence[str]] = None,
) -> None:
    if sample_paths is not None and len(sample_paths) != len(y_true):
        raise ValueError("sample_paths length must match the number of predictions.")
    fieldnames = [
        "sample_index",
        "sample_path",
        "true_index",
        "true_class",
        "pred_index",
        "pred_class",
    ]
    fieldnames.extend([f"prob_{i}_{name}" for i, name in enumerate(class_names)])

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(y_true)):
            row = {
                "sample_index": i,
                "sample_path": sample_paths[i] if sample_paths is not None else "",
                "true_index": int(y_true[i]),
                "true_class": class_names[int(y_true[i])],
                "pred_index": int(y_pred[i]),
                "pred_class": class_names[int(y_pred[i])],
            }
            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_idx}_{class_name}"] = float(y_prob[i, class_idx])
            writer.writerow(row)


def save_confusion_matrix_csv(
    cm: np.ndarray, class_names: Sequence[str], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name, *[int(value) for value in row]])


def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_stem: Path,
    normalize: bool,
) -> None:
    if normalize:
        denominator = cm.sum(axis=1, keepdims=True)
        matrix = np.divide(
            cm.astype(np.float64),
            denominator,
            out=np.zeros_like(cm, dtype=np.float64),
            where=denominator != 0,
        )
        title = "Normalized confusion matrix"
        value_format = ".2f"
    else:
        matrix = cm
        title = "Confusion matrix"
        value_format = "d"

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = format(value, value_format)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if float(value) > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    for suffix in (".pdf", ".eps", ".png"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_multiclass_roc_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_stem: Path,
) -> None:
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    fig, ax = plt.subplots(figsize=(8, 7))
    all_fpr = np.linspace(0.0, 1.0, 1001)
    interpolated_tprs = []

    for class_idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_prob[:, class_idx])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.3, label=f"{class_name} (AUC={class_auc:.3f})")
        interpolated_tprs.append(np.interp(all_fpr, fpr, tpr))

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr,
        mean_tpr,
        linestyle="--",
        linewidth=2.5,
        label=f"Macro-average (AUC={macro_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.2, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    for suffix in (".pdf", ".eps", ".png"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_path: Path, output_dir: Path) -> None:
    rows = []
    with history_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        return

    epochs = np.asarray([int(row["epoch"]) for row in rows])
    train_loss = np.asarray([float(row["train_loss"]) for row in rows])
    val_loss = np.asarray([float(row["val_loss"]) for row in rows])
    train_f1 = np.asarray([float(row["train_macro_f1"]) for row in rows])
    val_f1 = np.asarray([float(row["val_macro_f1"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="Training")
    ax.plot(epochs, val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".pdf", ".eps", ".png"):
        fig.savefig(output_dir / f"loss_curve{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_f1, label="Training")
    ax.plot(epochs, val_f1, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Training and validation Macro-F1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".pdf", ".eps", ".png"):
        fig.savefig(output_dir / f"macro_f1_curve{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def create_or_load_split(
    base_dataset: datasets.ImageFolder,
    val_ratio: float,
    seed: int,
    split_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(base_dataset.targets, dtype=np.int64)
    num_classes = len(base_dataset.classes)
    counts = np.bincount(targets, minlength=num_classes)

    if np.any(counts < 2):
        raise ValueError(
            "A stratified train/validation split requires at least two images in "
            f"every class. Class counts: {counts.tolist()}"
        )

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            split = json.load(f)
        current_sample_paths = [
            str(Path(path).relative_to(base_dataset.root))
            for path, _ in base_dataset.samples
        ]
        saved_sample_paths = split.get("sample_paths")
        if saved_sample_paths is not None and saved_sample_paths != current_sample_paths:
            raise RuntimeError(
                "The files in the training folder have changed since the split was "
                "created. Use a new --output_dir or delete the old split file."
            )
        if int(split.get("seed", seed)) != seed or not math.isclose(
            float(split.get("val_ratio", val_ratio)), val_ratio, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"Existing split file was created with seed={split.get('seed')} and "
                f"val_ratio={split.get('val_ratio')}, but the current run requests "
                f"seed={seed} and val_ratio={val_ratio}. Use a new --output_dir or "
                "delete the old split file."
            )
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        val_indices = np.asarray(split["val_indices"], dtype=np.int64)

        expected = set(range(len(base_dataset)))
        observed = set(train_indices.tolist()) | set(val_indices.tolist())
        if observed != expected or set(train_indices.tolist()) & set(val_indices.tolist()):
            raise RuntimeError(
                f"Existing split file is incompatible with the current dataset: {split_path}"
            )
        return train_indices, val_indices

    indices = np.arange(len(base_dataset), dtype=np.int64)

    requested_val_count = int(round(len(base_dataset) * val_ratio))
    val_count = max(num_classes, requested_val_count)
    if len(base_dataset) - val_count < num_classes:
        raise ValueError(
            "The training folder is too small to place at least one sample from "
            "every class in both the internal training and validation subsets. "
            f"Total={len(base_dataset)}, classes={num_classes}, "
            f"requested validation ratio={val_ratio}."
        )

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_count,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)

    save_json(
        {
            "seed": seed,
            "val_ratio": val_ratio,
            "actual_val_ratio": len(val_indices) / len(base_dataset),
            "num_total": len(base_dataset),
            "class_to_idx": base_dataset.class_to_idx,
            "sample_paths": [
                str(Path(path).relative_to(base_dataset.root))
                for path, _ in base_dataset.samples
            ],
            "train_indices": train_indices,
            "val_indices": val_indices,
        },
        split_path,
    )
    return train_indices, val_indices


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def save_dataset_summary(
    class_names: Sequence[str],
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    test_counts: np.ndarray,
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_index", "class_name", "train", "validation", "test"],
        )
        writer.writeheader()
        for idx, name in enumerate(class_names):
            writer.writerow(
                {
                    "class_index": idx,
                    "class_name": name,
                    "train": int(train_counts[idx]),
                    "validation": int(val_counts[idx]),
                    "test": int(test_counts[idx]),
                }
            )

def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    option: str,
    default: bool,
    help_text: Optional[str] = None,
) -> None:

    if not option.startswith("--"):
        raise ValueError("Boolean option must start with '--'.")

    option_name = option[2:]
    dest = option_name.replace("-", "_")
    negative_option = "--no-" + option_name

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        option,
        dest=dest,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        negative_option,
        dest=dest,
        action="store_false",
        help="Disable " + option + ".",
    )
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STFUEDATA ResNet-Laplacian-SemiLinear Transformer training"
    )

    parser.add_argument("--data_root", type=str, default="STFUEDATA")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="STFUE_RLST")
    parser.add_argument("--image_size", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.20)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--selection_metric",
        choices=["macro_f1", "balanced_accuracy", "accuracy"],
        default="macro_f1",
    )
    parser.add_argument(
        "--loss_type",
        choices=["ce", "weighted_ce", "focal"],
        default="weighted_ce",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    add_boolean_optional_argument(
        parser,
        "--train_hflip",
        default=True,
        help_text=(
            "Use random horizontal flipping during training. Disable with "
            "--no-train_hflip if horizontal flipping is not clinically valid."
        ),
    )
    parser.add_argument("--rotation_degrees", type=float, default=3.0)
    parser.add_argument(
        "--tta_mode",
        choices=["none", "hflip", "rot3", "hflip_rot3"],
        default="hflip",
        help="Pre-specify the TTA rule before final test evaluation.",
    )

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--classifier_dropout", type=float, default=0.20)
    parser.add_argument("--score_hidden_dim", type=int, default=16)
    parser.add_argument("--gamma_init", type=float, default=0.01)
    parser.add_argument("--gamma_max", type=float, default=1.0)
    parser.add_argument("--lap_alpha", type=float, default=0.20)
    parser.add_argument("--lap_alpha_max", type=float, default=0.50)
    add_boolean_optional_argument(
        parser,
        "--learnable_lap_alpha",
        default=True,
        help_text="Learn the bounded Laplacian enhancement coefficient.",
    )

    add_boolean_optional_argument(
        parser,
        "--deterministic",
        default=True,
        help_text="Use deterministic settings when supported.",
    )
    add_boolean_optional_argument(
        parser,
        "--amp",
        default=True,
        help_text="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=1000)

    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val_ratio must be between 0 and 1.")
    if args.embed_dim % args.num_heads != 0:
        parser.error("--embed_dim must be divisible by --num_heads.")
    if args.transformer_depth < 1:
        parser.error("--transformer_depth must be at least 1.")
    if args.num_heads < 1:
        parser.error("--num_heads must be at least 1.")
    if args.min_lr <= 0 or args.lr <= 0 or args.min_lr > args.lr:
        parser.error("Require 0 < min_lr <= lr.")
    if args.image_size < 64:
        parser.error("--image_size must be at least 64 for the current architecture.")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    pin_memory = device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_validation_model.pth"
    split_path = output_dir / "train_validation_split.json"
    history_path = output_dir / "training_history.csv"

    train_path = Path(args.data_root) / args.train_dir
    test_path = Path(args.data_root) / args.test_dir
    if not train_path.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {train_path}")
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_path}")

    base_train_dataset = datasets.ImageFolder(str(train_path))
    if len(base_train_dataset.classes) < 2:
        raise ValueError("At least two classes are required.")

    train_indices, val_indices = create_or_load_split(
        base_dataset=base_train_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_path=split_path,
    )

    mean, std = compute_mean_std(
        train_path=train_path,
        train_indices=train_indices,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    train_transform = build_train_transform(
        image_size=args.image_size,
        mean=mean,
        std=std,
        use_hflip=args.train_hflip,
        rotation_degrees=args.rotation_degrees,
    )
    eval_transform = build_eval_transform(args.image_size, mean, std)

    train_dataset_full = datasets.ImageFolder(str(train_path), transform=train_transform)
    val_dataset_full = datasets.ImageFolder(str(train_path), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(test_path), transform=eval_transform)

    if train_dataset_full.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError(
            "Class folders or their ordering differ between train and test.\n"
            f"Train mapping: {train_dataset_full.class_to_idx}\n"
            f"Test mapping:  {test_dataset.class_to_idx}"
        )

    train_dataset = Subset(train_dataset_full, train_indices.tolist())
    val_dataset = Subset(val_dataset_full, val_indices.tolist())

    class_names = train_dataset_full.classes
    num_classes = len(class_names)
    targets = np.asarray(base_train_dataset.targets, dtype=np.int64)
    train_counts = np.bincount(targets[train_indices], minlength=num_classes)
    val_counts = np.bincount(targets[val_indices], minlength=num_classes)
    test_targets = np.asarray(test_dataset.targets, dtype=np.int64)
    test_counts = np.bincount(test_targets, minlength=num_classes)
    test_sample_paths = [str(Path(path)) for path, _ in test_dataset.samples]

    if np.any(test_counts == 0):
        raise ValueError(
            f"Every class must appear in the fixed test set. Test counts: {test_counts.tolist()}"
        )

    save_dataset_summary(
        class_names,
        train_counts,
        val_counts,
        test_counts,
        output_dir / "dataset_distribution.csv",
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed + 1,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed + 2,
    )

    model = ResNetLaplacianSemiLinearTransformer(
        num_classes=num_classes,
        in_channels=1,
        embed_dim=args.embed_dim,
        transformer_depth=args.transformer_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        score_hidden_dim=args.score_hidden_dim,
        gamma_init=args.gamma_init,
        gamma_max=args.gamma_max,
        lap_alpha=args.lap_alpha,
        learnable_lap_alpha=args.learnable_lap_alpha,
        lap_alpha_max=args.lap_alpha_max,
        classifier_dropout=args.classifier_dropout,
        pos_base_size=7,
    ).to(device)

    class_weights = compute_class_weights(train_counts)
    criterion = build_criterion(
        loss_type=args.loss_type,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        device=device,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr / args.lr,
    )
    scaler = make_grad_scaler(amp_enabled)

    save_json(
        {
            "args": vars(args),
            "device": str(device),
            "class_to_idx": train_dataset_full.class_to_idx,
            "class_names": class_names,
            "normalization": {"mean": mean, "std": std},
            "class_weights": class_weights,
            "parameter_count": count_all_parameters(model),
        },
        output_dir / "experiment_config.json",
    )

    history_writer = HistoryWriter(history_path)
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
        )

        val_result, _, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            tta_mode="none",
            amp_enabled=amp_enabled,
            description="Validation",
        )

        score = metric_value(val_result, args.selection_metric)
        elapsed = time.time() - epoch_start

        history_writer.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "train_macro_f1": train_result.macro_f1,
                "train_balanced_accuracy": train_result.balanced_accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "val_macro_f1": val_result.macro_f1,
                "val_balanced_accuracy": val_result.balanced_accuracy,
                "elapsed_seconds": elapsed,
            }
        )

        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_validation_score": best_score,
                    "selection_metric": args.selection_metric,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": train_dataset_full.class_to_idx,
                    "class_names": class_names,
                    "normalization": {"mean": mean, "std": std},
                    "model_config": {
                        "num_classes": num_classes,
                        "in_channels": 1,
                        "embed_dim": args.embed_dim,
                        "transformer_depth": args.transformer_depth,
                        "num_heads": args.num_heads,
                        "mlp_ratio": args.mlp_ratio,
                        "dropout": args.dropout,
                        "score_hidden_dim": args.score_hidden_dim,
                        "gamma_init": args.gamma_init,
                        "gamma_max": args.gamma_max,
                        "lap_alpha": args.lap_alpha,
                        "learnable_lap_alpha": args.learnable_lap_alpha,
                        "lap_alpha_max": args.lap_alpha_max,
                        "classifier_dropout": args.classifier_dropout,
                        "pos_base_size": 7,
                    },
                    "args": vars(args),
                    "validation_metrics": asdict(val_result),
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:

            break

    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved during training.")

    save_training_curves(history_path, output_dir)

    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_result_no_tta, y_true, y_pred_no_tta, y_prob_no_tta = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        tta_mode="none",
        amp_enabled=amp_enabled,
        description="Final test (no TTA)",
    )
    metrics_no_tta, cm_no_tta, per_class_no_tta = multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred_no_tta,
        y_prob=y_prob_no_tta,
        class_names=class_names,
        loss=test_result_no_tta.loss,
    )

    ci_no_tta = stratified_bootstrap_confidence_intervals(
        y_true=y_true,
        y_pred=y_pred_no_tta,
        y_prob=y_prob_no_tta,
        num_classes=num_classes,
        iterations=args.bootstrap_iters,
        seed=args.seed + 2025,
    )
    metrics_no_tta["bootstrap_95_ci"] = ci_no_tta

    save_json(metrics_no_tta, output_dir / "test_metrics_no_tta.json")
    save_per_class_metrics(
        per_class_no_tta, output_dir / "test_per_class_metrics_no_tta.csv"
    )
    save_predictions(
        y_true,
        y_pred_no_tta,
        y_prob_no_tta,
        class_names,
        output_dir / "test_predictions_no_tta.csv",
        sample_paths=test_sample_paths,
    )
    save_confusion_matrix_csv(
        cm_no_tta, class_names, output_dir / "confusion_matrix_no_tta.csv"
    )
    save_confusion_matrix_figure(
        cm_no_tta,
        class_names,
        output_dir / "confusion_matrix_no_tta",
        normalize=False,
    )
    save_confusion_matrix_figure(
        cm_no_tta,
        class_names,
        output_dir / "confusion_matrix_no_tta_normalized",
        normalize=True,
    )
    save_multiclass_roc_figure(
        y_true,
        y_prob_no_tta,
        class_names,
        output_dir / "roc_curves_no_tta",
    )

    final_results = {
        "best_validation_epoch": best_epoch,
        "best_validation_score": best_score,
        "selection_metric": args.selection_metric,
        "no_tta": metrics_no_tta,
        "learned_parameters": model.get_interpretable_parameters(),
    }

    if args.tta_mode != "none":
        test_result_tta, y_true_tta, y_pred_tta, y_prob_tta = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            tta_mode=args.tta_mode,
            amp_enabled=amp_enabled,
            description=f"Final test (TTA: {args.tta_mode})",
        )
        if not np.array_equal(y_true, y_true_tta):
            raise RuntimeError("Test label order changed between no-TTA and TTA evaluation.")

        metrics_tta, cm_tta, per_class_tta = multiclass_metrics(
            y_true=y_true_tta,
            y_pred=y_pred_tta,
            y_prob=y_prob_tta,
            class_names=class_names,
            loss=test_result_tta.loss,
        )

        ci_tta = stratified_bootstrap_confidence_intervals(
            y_true=y_true_tta,
            y_pred=y_pred_tta,
            y_prob=y_prob_tta,
            num_classes=num_classes,
            iterations=args.bootstrap_iters,
            seed=args.seed + 2026,
        )
        metrics_tta["bootstrap_95_ci"] = ci_tta

        save_json(metrics_tta, output_dir / f"test_metrics_tta_{args.tta_mode}.json")
        save_per_class_metrics(
            per_class_tta,
            output_dir / f"test_per_class_metrics_tta_{args.tta_mode}.csv",
        )
        save_predictions(
            y_true_tta,
            y_pred_tta,
            y_prob_tta,
            class_names,
            output_dir / f"test_predictions_tta_{args.tta_mode}.csv",
            sample_paths=test_sample_paths,
        )
        save_confusion_matrix_csv(
            cm_tta,
            class_names,
            output_dir / f"confusion_matrix_tta_{args.tta_mode}.csv",
        )
        save_confusion_matrix_figure(
            cm_tta,
            class_names,
            output_dir / f"confusion_matrix_tta_{args.tta_mode}",
            normalize=False,
        )
        save_confusion_matrix_figure(
            cm_tta,
            class_names,
            output_dir / f"confusion_matrix_tta_{args.tta_mode}_normalized",
            normalize=True,
        )
        save_multiclass_roc_figure(
            y_true_tta,
            y_prob_tta,
            class_names,
            output_dir / f"roc_curves_tta_{args.tta_mode}",
        )
        final_results[f"tta_{args.tta_mode}"] = metrics_tta
    else:
        final_results["no_tta"] = metrics_no_tta

    save_json(final_results, output_dir / "final_results.json")