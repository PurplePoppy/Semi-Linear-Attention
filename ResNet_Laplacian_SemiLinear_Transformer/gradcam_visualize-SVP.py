from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

CHECKPOINT_PATH = Path(
    "/data/coding/STFUE_RLST/best_validation_model.pth"
)

IMAGE_PATH = Path(
    "/data/coding/SVP/SVP-4.jpg"
)

OUTPUT_ROOT = Path(
    "/data/coding/SVP4_final_laplacian"
)

SVP_PREFIX = "SVP"

TARGET_LAYER_NAME = "laplacian"

MAX_OVERLAY_ALPHA = 0.55

OVERLAY_GAMMA = 1.50

FIGURE_DPI = 600

DEVICE_MODE = "auto"

CAM_COLORMAP = "jet"

def safe_torch_load(
    path: Path,
    device: torch.device,
):

    try:

        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=device,
        )


def get_device() -> torch.device:

    if DEVICE_MODE == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if DEVICE_MODE == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but CUDA is unavailable."
            )

        return torch.device(
            "cuda"
        )

    if DEVICE_MODE == "cpu":

        return torch.device(
            "cpu"
        )

    raise ValueError(
        f"Unknown DEVICE_MODE: {DEVICE_MODE}"
    )


def normalize_cam(
    cam: np.ndarray,
) -> np.ndarray:

    cam = np.asarray(
        cam,
        dtype=np.float32,
    )

    cam = np.maximum(
        cam,
        0.0,
    )

    minimum = float(
        cam.min()
    )

    maximum = float(
        cam.max()
    )

    if maximum - minimum > 1e-12:

        cam = (
            cam - minimum
        ) / (
            maximum - minimum
        )

    else:

        cam = np.zeros_like(
            cam,
            dtype=np.float32,
        )

    return np.clip(
        cam,
        0.0,
        1.0,
    )

class ResizeWithPad:

    def __init__(
        self,
        size: int,
        fill: int = 0,
        interpolation:
            InterpolationMode =
            InterpolationMode.BILINEAR,
    ):

        self.size = int(
            size
        )

        self.fill = fill

        self.interpolation = (
            interpolation
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:

        width, height = (
            image.size
        )

        if width <= 0 or height <= 0:

            raise ValueError(
                f"Invalid image size: "
                f"{(width, height)}"
            )

        scale = (
            self.size
            / max(
                width,
                height,
            )
        )

        new_width = max(
            1,
            int(
                round(
                    width
                    * scale
                )
            ),
        )

        new_height = max(
            1,
            int(
                round(
                    height
                    * scale
                )
            ),
        )

        image = TF.resize(
            image,
            [
                new_height,
                new_width,
            ],
            interpolation=self.interpolation,
            antialias=True,
        )

        pad_left = (
            self.size
            - new_width
        ) // 2

        pad_right = (
            self.size
            - new_width
            - pad_left
        )

        pad_top = (
            self.size
            - new_height
        ) // 2

        pad_bottom = (
            self.size
            - new_height
            - pad_top
        )

        return TF.pad(
            image,
            [
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            fill=self.fill,
            padding_mode="constant",
        )


def build_eval_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
):

    return transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=1
            ),

            ResizeWithPad(
                image_size
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if (
            stride != 1
            or
            in_channels != out_channels
        ):

            self.shortcut = (
                nn.Sequential(

                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),

                    nn.BatchNorm2d(
                        out_channels
                    ),
                )
            )

        else:

            self.shortcut = (
                nn.Identity()
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = self.shortcut(
            x
        )

        out = F.relu(
            self.bn1(
                self.conv1(
                    x
                )
            ),
            inplace=True,
        )

        out = self.bn2(
            self.conv2(
                out
            )
        )

        out = F.relu(
            out + identity,
            inplace=True,
        )

        return out

class LaplacianEnhancement(nn.Module):

    def __init__(
        self,
        channels: int,
        alpha_init: float = 0.2,
        learnable_alpha: bool = True,
        alpha_max: float = 0.5,
    ):

        super().__init__()

        if not 0 <= alpha_init < alpha_max:

            raise ValueError(
                "alpha_init must satisfy "
                "0 <= alpha_init < alpha_max"
            )

        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        kernel = kernel.view(
            1,
            1,
            3,
            3,
        ).repeat(
            channels,
            1,
            1,
            1,
        )

        self.register_buffer(
            "weight",
            kernel,
        )

        self.channels = (
            channels
        )

        self.learnable_alpha = (
            learnable_alpha
        )

        self.alpha_max = float(
            alpha_max
        )

        if learnable_alpha:

            ratio = min(
                max(
                    alpha_init
                    / alpha_max,
                    1e-6,
                ),
                1.0 - 1e-6,
            )

            raw = math.log(
                ratio
                / (
                    1.0 - ratio
                )
            )

            self.alpha_raw = (
                nn.Parameter(
                    torch.tensor(
                        raw,
                        dtype=torch.float32,
                    )
                )
            )

        else:

            self.register_buffer(
                "alpha_fixed",
                torch.tensor(
                    alpha_init,
                    dtype=torch.float32,
                ),
            )

    def current_alpha(
        self,
    ):

        if self.learnable_alpha:

            return (
                self.alpha_max
                * torch.sigmoid(
                    self.alpha_raw
                )
            )

        return (
            self.alpha_fixed
        )

    def forward(
        self,
        x,
    ):

        x_pad = F.pad(
            x,
            (
                1,
                1,
                1,
                1,
            ),
            mode="reflect",
        )

        lap = F.conv2d(
            x_pad,
            self.weight,
            stride=1,
            padding=0,
            groups=self.channels,
        )

        return (
            x
            +
            self.current_alpha()
            * lap
        )

class SemiLinearAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        score_hidden_dim: int,
        gamma_init: float,
        gamma_max: float,
    ):

        super().__init__()

        if (
            embed_dim
            % num_heads
            != 0
        ):

            raise ValueError(
                "embed_dim must be divisible "
                "by num_heads"
            )

        self.embed_dim = (
            embed_dim
        )

        self.num_heads = (
            num_heads
        )

        self.head_dim = (
            embed_dim
            // num_heads
        )

        self.scale = (
            self.head_dim
            ** -0.5
        )

        self.gamma_max = float(
            gamma_max
        )

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
            bias=True,
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.score_mlp = (
            nn.Sequential(

                nn.Linear(
                    1,
                    score_hidden_dim,
                ),

                nn.GELU(),

                nn.Linear(
                    score_hidden_dim,
                    1,
                ),
            )
        )

        init_ratio = (
            gamma_init
            / gamma_max
        )

        raw_init = (
            0.5
            * math.log(
                (
                    1 + init_ratio
                )
                /
                (
                    1 - init_ratio
                )
            )
        )

        self.gamma_raw = (
            nn.Parameter(
                torch.full(
                    (
                        num_heads,
                        1,
                        1,
                    ),
                    raw_init,
                    dtype=torch.float32,
                )
            )
        )

        self.attn_drop = (
            nn.Dropout(
                dropout
            )
        )

        self.proj_drop = (
            nn.Dropout(
                dropout
            )
        )

    def current_gamma(
        self,
    ):

        return (
            self.gamma_max
            * torch.tanh(
                self.gamma_raw
            )
        )

    def forward(
        self,
        x,
    ):

        batch_size, num_tokens, channels = (
            x.shape
        )

        qkv = self.qkv(
            x
        )

        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(
            2,
            0,
            3,
            1,
            4,
        )

        q, k, v = qkv.unbind(
            dim=0
        )

        scores = (
            torch.matmul(
                q,
                k.transpose(
                    -2,
                    -1,
                ),
            )
            * self.scale
        )

        nonlinear_scores = (
            self.score_mlp(
                scores.unsqueeze(
                    -1
                )
            ).squeeze(
                -1
            )
        )

        semi_scores = (
            scores
            +
            self.current_gamma()
            .unsqueeze(
                0
            )
            * nonlinear_scores
        )

        attention = torch.softmax(
            semi_scores,
            dim=-1,
        )

        attention = (
            self.attn_drop(
                attention
            )
        )

        out = torch.matmul(
            attention,
            v,
        )

        out = out.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            num_tokens,
            channels,
        )

        out = self.out_proj(
            out
        )

        return self.proj_drop(
            out
        )

class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        dropout: float,
    ):

        super().__init__()

        hidden_dim = int(
            round(
                embed_dim
                * mlp_ratio
            )
        )

        self.net = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embed_dim,
            ),

            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.net(
            x
        )

class SemiLinearTransformerBlock(
    nn.Module
):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        score_hidden_dim: int,
        gamma_init: float,
        gamma_max: float,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = (
            SemiLinearAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                score_hidden_dim=(
                    score_hidden_dim
                ),
                gamma_init=(
                    gamma_init
                ),
                gamma_max=(
                    gamma_max
                ),
            )
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.ffn = FeedForward(
            embed_dim,
            mlp_ratio,
            dropout,
        )

    def forward(
        self,
        x,
    ):

        x = (
            x
            +
            self.attention(
                self.norm1(
                    x
                )
            )
        )

        x = (
            x
            +
            self.ffn(
                self.norm2(
                    x
                )
            )
        )

        return x

class ResNetLaplacianSemiLinearTransformer(
    nn.Module
):

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
    ):

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

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self._current_channels = (
            64
        )

        self.layer1 = self._make_layer(
            64,
            2,
            1,
        )

        self.layer2 = self._make_layer(
            128,
            2,
            2,
        )

        self.layer3 = self._make_layer(
            256,
            2,
            2,
        )

        self.layer4 = self._make_layer(
            512,
            2,
            2,
        )

        self.token_projection = (
            nn.Sequential(

                nn.Conv2d(
                    512,
                    embed_dim,
                    kernel_size=1,
                    bias=False,
                ),

                nn.BatchNorm2d(
                    embed_dim
                ),

                nn.GELU(),
            )
        )

        self.laplacian = (
            LaplacianEnhancement(
                channels=embed_dim,
                alpha_init=lap_alpha,
                learnable_alpha=(
                    learnable_lap_alpha
                ),
                alpha_max=(
                    lap_alpha_max
                ),
            )
        )

        self.pos_embed_2d = nn.Parameter(
            torch.zeros(
                1,
                embed_dim,
                pos_base_size,
                pos_base_size,
            )
        )

        self.transformer = (
            nn.Sequential(
                *[
                    SemiLinearTransformerBlock(
                        embed_dim=(
                            embed_dim
                        ),
                        num_heads=(
                            num_heads
                        ),
                        mlp_ratio=(
                            mlp_ratio
                        ),
                        dropout=(
                            dropout
                        ),
                        score_hidden_dim=(
                            score_hidden_dim
                        ),
                        gamma_init=(
                            gamma_init
                        ),
                        gamma_max=(
                            gamma_max
                        ),
                    )

                    for _ in range(
                        transformer_depth
                    )
                ]
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.classifier_dropout = (
            nn.Dropout(
                classifier_dropout
            )
        )

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

    def _make_layer(
        self,
        out_channels,
        blocks,
        stride,
    ):

        layers = [

            BasicBlock(
                self._current_channels,
                out_channels,
                stride,
            )
        ]

        self._current_channels = (
            out_channels
        )

        for _ in range(
            1,
            blocks,
        ):

            layers.append(

                BasicBlock(
                    self._current_channels,
                    out_channels,
                    1,
                )
            )

        return nn.Sequential(
            *layers
        )

    def forward_features(
        self,
        x,
    ):

        x = self.stem(
            x
        )

        x = self.layer1(
            x
        )

        x = self.layer2(
            x
        )

        x = self.layer3(
            x
        )

        x = self.layer4(
            x
        )

        x = self.token_projection(
            x
        )

        x = self.laplacian(
            x
        )

        _, _, height, width = (
            x.shape
        )

        pos_embed = F.interpolate(
            self.pos_embed_2d,
            size=(
                height,
                width,
            ),
            mode="bicubic",
            align_corners=False,
        )

        x = (
            x
            + pos_embed
        )

        x = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        x = self.transformer(
            x
        )

        x = self.norm(
            x
        )

        return x.mean(
            dim=1
        )

    def forward(
        self,
        x,
    ):

        features = (
            self.forward_features(
                x
            )
        )

        features = (
            self.classifier_dropout(
                features
            )
        )

        return self.classifier(
            features
        )

class GradCAM:

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):

        self.model = (
            model
        )

        self.activations = (
            None
        )

        self.gradients = (
            None
        )

        self.handle = (
            target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        if output.ndim != 4:

            raise RuntimeError(
                "Grad-CAM target layer "
                "must output B,C,H,W. "
                f"Got: {output.shape}"
            )

        self.activations = (
            output
        )

        output.register_hook(
            self._save_gradient
        )

    def _save_gradient(
        self,
        gradient,
    ):

        self.gradients = (
            gradient.detach()
        )

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        self.activations = (
            None
        )

        self.gradients = (
            None
        )

        logits = self.model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_class = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        target_score = logits[
            0,
            target_class,
        ]

        target_score.backward()

        if self.activations is None:

            raise RuntimeError(
                "No Grad-CAM activation captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "No Grad-CAM gradient captured."
            )

        activations = (
            self.activations
            .detach()
        )

        gradients = (
            self.gradients
        )

        weights = gradients.mean(
            dim=(
                2,
                3,
            ),
            keepdim=True,
        )

        cam = (
            weights
            * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam_native = (
            cam[
                0,
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        cam_native = normalize_cam(
            cam_native
        )

        probabilities_np = (
            probabilities[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        return (
            cam_native,
            predicted_class,
            probabilities_np,
        )

    def close(
        self,
    ):

        self.handle.remove()

def get_target_layer(
    model,
):

    if TARGET_LAYER_NAME == "laplacian":

        return (
            model.laplacian
        )

    raise ValueError(
        "The final version only supports "
        "TARGET_LAYER_NAME='laplacian'."
    )

def resize_cam_bicubic(
    cam: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:

    cam = normalize_cam(
        cam
    )

    cam_uint8 = (
        np.clip(
            cam,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    pil_cam = Image.fromarray(
        cam_uint8,
        mode="L",
    )

    if hasattr(
        Image,
        "Resampling",
    ):

        resample = (
            Image.Resampling.BICUBIC
        )

    else:

        resample = (
            Image.BICUBIC
        )

    pil_cam = pil_cam.resize(
        (
            target_width,
            target_height,
        ),
        resample=resample,
    )

    resized = (
        np.asarray(
            pil_cam,
            dtype=np.float32,
        )
        / 255.0
    )

    return normalize_cam(
        resized
    )


def restore_cam_to_original(
    cam_native: np.ndarray,
    original_size,
    image_size: int,
) -> np.ndarray:

    original_width, original_height = (
        original_size
    )

    cam_square = resize_cam_bicubic(
        cam=cam_native,
        target_width=image_size,
        target_height=image_size,
    )

    scale = (
        image_size
        / max(
            original_width,
            original_height,
        )
    )

    resized_width = max(
        1,
        int(
            round(
                original_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                original_height
                * scale
            )
        ),
    )

    pad_left = (
        image_size
        - resized_width
    ) // 2

    pad_top = (
        image_size
        - resized_height
    ) // 2

    cam_unpadded = cam_square[
        pad_top:
        pad_top + resized_height,

        pad_left:
        pad_left + resized_width,
    ]

    cam_original = resize_cam_bicubic(
        cam=cam_unpadded,
        target_width=original_width,
        target_height=original_height,
    )

    return normalize_cam(
        cam_original
    )

def create_heatmap(
    cam: np.ndarray,
) -> np.ndarray:

    cmap = plt.get_cmap(
        CAM_COLORMAP
    )

    rgba = cmap(
        np.clip(
            cam,
            0.0,
            1.0,
        )
    )

    rgb = rgba[
        ...,
        :3
    ]

    return rgb.astype(
        np.float32
    )


def create_adaptive_alpha_map(
    cam: np.ndarray,
) -> np.ndarray:

    cam = np.clip(
        cam,
        0.0,
        1.0,
    )

    alpha_map = (
        MAX_OVERLAY_ALPHA
        * np.power(
            cam,
            OVERLAY_GAMMA,
        )
    )

    return np.clip(
        alpha_map,
        0.0,
        MAX_OVERLAY_ALPHA,
    ).astype(
        np.float32
    )


def create_adaptive_overlay(
    gray: np.ndarray,
    cam: np.ndarray,
):

    original_rgb = np.stack(
        [
            gray,
            gray,
            gray,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    heatmap = create_heatmap(
        cam
    )

    alpha_map = (
        create_adaptive_alpha_map(
            cam
        )
    )

    alpha_rgb = alpha_map[
        ...,
        None
    ]

    overlay = (
        (
            1.0
            - alpha_rgb
        )
        * original_rgb
        +
        alpha_rgb
        * heatmap
    )

    overlay = np.clip(
        overlay,
        0.0,
        1.0,
    )

    return (
        original_rgb,
        heatmap,
        overlay,
        alpha_map,
    )


def save_rgb(
    image: np.ndarray,
    path: Path,
):

    array = (
        np.clip(
            image,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    Image.fromarray(
        array
    ).save(
        path
    )


def save_gray(
    image: np.ndarray,
    path: Path,
):

    array = (
        np.clip(
            image,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    Image.fromarray(
        array,
        mode="L",
    ).save(
        path
    )

def save_diagnostic_figure(
    original_rgb,
    cam,
    heatmap,
    alpha_map,
    overlay,
):

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            15.5,
            3.25,
        ),
    )

    axes[0].imshow(
        original_rgb
    )

    axes[0].set_title(
        "Original",
        fontsize=11,
    )

    axes[1].imshow(
        cam,
        cmap=CAM_COLORMAP,
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].set_title(
        "Laplacian Grad-CAM",
        fontsize=11,
    )

    axes[2].imshow(
        heatmap
    )

    axes[2].set_title(
        "Color Heatmap",
        fontsize=11,
    )

    alpha_display = (
        alpha_map
        / MAX_OVERLAY_ALPHA
    )

    axes[3].imshow(
        alpha_display,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axes[3].set_title(
        "Adaptive Alpha",
        fontsize=11,
    )

    axes[4].imshow(
        overlay
    )

    axes[4].set_title(
        "Final Overlay",
        fontsize=11,
    )

    for ax in axes:

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )

        for spine in (
            ax.spines.values()
        ):

            spine.set_visible(
                False
            )

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.88,
        bottom=0.01,
        wspace=0.025,
    )

    fig.savefig(
        OUTPUT_ROOT
        / "SVP4_diagnostic_FINAL.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        OUTPUT_ROOT
        / "SVP4_diagnostic_FINAL.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(
        fig
    )

def save_publication_figure(
    original_rgb,
    cam,
    overlay,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            9.2,
            3.05,
        ),
    )

    axes[0].imshow(
        original_rgb
    )

    axes[0].set_title(
        "(a) Original",
        fontsize=12,
        pad=5,
    )

    axes[1].imshow(
        cam,
        cmap=CAM_COLORMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
    )

    axes[1].set_title(
        "(b) Grad-CAM",
        fontsize=12,
        pad=5,
    )

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        "(c) Overlay",
        fontsize=12,
        pad=5,
    )

    for ax in axes:

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )

        for spine in (
            ax.spines.values()
        ):

            spine.set_visible(
                False
            )

    fig.subplots_adjust(
        left=0.003,
        right=0.997,
        top=0.89,
        bottom=0.005,
        wspace=0.018,
    )

    fig.savefig(
        OUTPUT_ROOT
        / "SVP4_GradCAM_FINAL.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.015,
    )

    fig.savefig(
        OUTPUT_ROOT
        / "SVP4_GradCAM_FINAL.pdf",
        bbox_inches="tight",
        pad_inches=0.015,
    )

    fig.savefig(
        OUTPUT_ROOT
        / "SVP4_GradCAM_FINAL.eps",
        bbox_inches="tight",
        pad_inches=0.015,
    )

    plt.close(
        fig
    )

def main():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            "SVP-4 image not found:\n"
            f"{IMAGE_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    if device.type == "cuda":

    checkpoint = safe_torch_load(
        CHECKPOINT_PATH,
        device,
    )

    required_keys = [
        "model_state_dict",
        "model_config",
        "class_names",
        "normalization",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(
                "Checkpoint missing key: "
                f"{key}"
            )

    class_names = list(
        checkpoint[
            "class_names"
        ]
    )

    model_config = (
        checkpoint[
            "model_config"
        ]
    )

    normalization = (
        checkpoint[
            "normalization"
        ]
    )

    mean = (
        normalization[
            "mean"
        ]
    )

    std = (
        normalization[
            "std"
        ]
    )

    image_size = int(
        checkpoint.get(
            "args",
            {},
        ).get(
            "image_size",
            200,
        )
    )

    for index, name in enumerate(
        class_names
    ):

    svp_candidates = [

        (
            index,
            name,
        )

        for index, name
        in enumerate(
            class_names
        )

        if name.upper().startswith(
            SVP_PREFIX.upper()
        )
    ]

    if not svp_candidates:

        raise RuntimeError(
            "SVP class was not found "
            "in checkpoint class_names."
        )

    svp_index, svp_name = (
        svp_candidates[
            0
        ]
    )

    model = (
        ResNetLaplacianSemiLinearTransformer(
            **model_config
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    learned_alpha = float(
        model.laplacian
        .current_alpha()
        .detach()
        .cpu()
        .item()
    )

    transform = build_eval_transform(
        image_size=image_size,
        mean=mean,
        std=std,
    )

    image = Image.open(
        IMAGE_PATH
    ).convert(
        "L"
    )

    tensor = (
        transform(
            image
        )
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )

    gray = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    target_layer = (
        get_target_layer(
            model
        )
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    (
        cam_native,
        predicted_class,
        probabilities,
    ) = gradcam.generate(
        image_tensor=tensor,
        target_class=svp_index,
    )

    gradcam.close()

    predicted_name = (
        class_names[
            predicted_class
        ]
    )

    svp_probability = float(
        probabilities[
            svp_index
        ]
    )

    predicted_probability = float(
        probabilities[
            predicted_class
        ]
    )

    is_correct = (
        predicted_class
        == svp_index
    )

    cam_original = (
        restore_cam_to_original(
            cam_native=cam_native,
            original_size=image.size,
            image_size=image_size,
        )
    )

    (
        original_rgb,
        heatmap_rgb,
        overlay_rgb,
        alpha_map,
    ) = create_adaptive_overlay(
        gray=gray,
        cam=cam_original,
    )

    np.save(
        OUTPUT_ROOT
        / "SVP4_cam_native_raw.npy",
        cam_native,
    )

    np.save(
        OUTPUT_ROOT
        / "SVP4_cam_original_raw.npy",
        cam_original,
    )

    np.save(
        OUTPUT_ROOT
        / "SVP4_adaptive_alpha_raw.npy",
        alpha_map,
    )

    save_rgb(
        original_rgb,
        OUTPUT_ROOT
        / "SVP4_original.png",
    )

    save_gray(
        cam_original,
        OUTPUT_ROOT
        / "SVP4_gradcam_gray.png",
    )

    save_rgb(
        heatmap_rgb,
        OUTPUT_ROOT
        / "SVP4_gradcam_heatmap.png",
    )

    save_gray(
        alpha_map
        / MAX_OVERLAY_ALPHA,
        OUTPUT_ROOT
        / "SVP4_adaptive_alpha.png",
    )

    save_rgb(
        overlay_rgb,
        OUTPUT_ROOT
        / "SVP4_overlay_FINAL.png",
    )

    save_diagnostic_figure(
        original_rgb=original_rgb,
        cam=cam_original,
        heatmap=heatmap_rgb,
        alpha_map=alpha_map,
        overlay=overlay_rgb,
    )

    save_publication_figure(
        original_rgb=original_rgb,
        cam=cam_original,
        overlay=overlay_rgb,
    )

    result_txt = (
        OUTPUT_ROOT
        / "SVP4_result.txt"
    )

    with result_txt.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Final SVP-4 Laplacian Grad-CAM\n"
        )

        file.write(
            "=" * 60
            + "\n"
        )

        file.write(
            f"Image: {IMAGE_PATH}\n"
        )

        file.write(
            f"Checkpoint: "
            f"{CHECKPOINT_PATH}\n"
        )

        file.write(
            f"Target layer: "
            f"model.laplacian\n"
        )

        file.write(
            f"Ground truth: "
            f"{svp_name}\n"
        )

        file.write(
            f"Predicted class: "
            f"{predicted_name}\n"
        )

        file.write(
            f"Correct: "
            f"{is_correct}\n"
        )

        file.write(
            f"SVP probability: "
            f"{svp_probability:.8f}\n"
        )

        file.write(
            f"Predicted probability: "
            f"{predicted_probability:.8f}\n"
        )

        file.write(
            f"Laplacian alpha: "
            f"{learned_alpha:.8f}\n"
        )

        file.write(
            f"Model input size: "
            f"{image_size}\n"
        )

        file.write(
            f"Original image size: "
            f"{image.size}\n"
        )

        file.write(
            f"Native CAM size: "
            f"{cam_native.shape}\n"
        )

        file.write(
            f"Interpolation: bicubic\n"
        )

        file.write(
            f"Overlay method: "
            f"adaptive alpha blending\n"
        )

        file.write(
            f"Maximum overlay alpha: "
            f"{MAX_OVERLAY_ALPHA}\n"
        )

        file.write(
            f"Overlay gamma: "
            f"{OVERLAY_GAMMA}\n"
        )

        file.write(
            "Multi-layer fusion: False\n"
        )

        file.write(
            "Artificial hotspot thresholding: False\n"
        )

        file.write(
            "Artificial hotspot relocation: False\n"
        )

        file.write(
            "Gaussian CAM smoothing: False\n"
        )