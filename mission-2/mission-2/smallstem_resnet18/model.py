"""Mission 2 log-Mel 화자 역할 분류 모델."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpecAugment(nn.Module):
    def __init__(self, freq_masks: int = 2, time_masks: int = 2, max_freq: int = 8, max_time: int = 4):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.max_freq = max_freq
        self.max_time = max_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = x.clone()
        batch, _, freq_bins, time_bins = x.shape
        freq_axis = torch.arange(freq_bins, device=x.device).view(1, freq_bins)
        time_axis = torch.arange(time_bins, device=x.device).view(1, time_bins)
        for _ in range(self.freq_masks):
            widths = torch.randint(0, min(self.max_freq, freq_bins) + 1, (batch, 1), device=x.device)
            starts = (torch.rand(batch, 1, device=x.device) * (freq_bins - widths + 1)).floor().long()
            mask = (freq_axis >= starts) & (freq_axis < starts + widths)
            x = x.masked_fill(mask[:, None, :, None], 0.0)
        for _ in range(self.time_masks):
            widths = torch.randint(0, min(self.max_time, time_bins) + 1, (batch, 1), device=x.device)
            starts = (torch.rand(batch, 1, device=x.device) * (time_bins - widths + 1)).floor().long()
            mask = (time_axis >= starts) & (time_axis < starts + widths)
            x = x.masked_fill(mask[:, None, None, :], 0.0)
        return x


class SpeakerResNet18(nn.Module):
    """작은 Mel 입력에도 동작하는 ImageNet-pretrained ResNet18."""

    def __init__(self, pretrained: bool = False, dropout: float = 0.30, spec_augment: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.augment = SpecAugment() if spec_augment else nn.Identity()
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.image_mean) / self.image_std
        return self.backbone(x)


def _normalized_rgb(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = x.repeat(1, 3, 1, 1)
    return (x - mean) / std


def _small_stem_resnet18(pretrained: bool) -> nn.Module:
    """64x24 입력을 너무 빨리 축소하지 않는 ResNet18 backbone."""
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    old_weight = backbone.conv1.weight.detach()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained:
        # 7x7 ImageNet kernel 전체를 3x3으로 축소해 stem 초기값으로 재사용한다.
        resized = F.interpolate(
            old_weight.reshape(-1, 1, 7, 7),
            size=(3, 3),
            mode="bilinear",
            align_corners=False,
        ).reshape(64, 3, 3, 3)
        with torch.no_grad():
            backbone.conv1.weight.copy_(resized)
    backbone.maxpool = nn.Identity()
    return backbone


class SmallStemSpeakerResNet18(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.25, spec_augment: bool = True):
        super().__init__()
        self.augment = SpecAugment(max_freq=6, max_time=3) if spec_augment else nn.Identity()
        self.backbone = _small_stem_resnet18(pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        return self.backbone(_normalized_rgb(x, self.image_mean, self.image_std))


class TDNNResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        hidden = max(16, channels // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.layers(x)
        residual = residual * self.se(residual)
        return self.activation(x + residual)


class TDNNEncoder(nn.Module):
    """Mel-bin을 채널, frame을 시간축으로 처리하는 화자 특징 encoder."""

    def __init__(self, n_mels: int = 64, channels: int = 192, embedding_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [TDNNResidualBlock(channels, dilation=value) for value in (1, 2, 3)]
        )
        self.merge = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 96, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(96, 1, kernel_size=1),
        )
        self.projection = nn.Sequential(
            nn.Linear(channels * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        x = self.merge(torch.cat(outputs, dim=1))
        weights = self.attention(x).softmax(dim=2)
        mean = torch.sum(weights * x, dim=2)
        second_moment = torch.sum(weights * x.square(), dim=2)
        std = (second_moment - mean.square()).clamp_min(1e-5).sqrt()
        return self.projection(torch.cat([mean, std], dim=1))


class HybridSpeakerNet(nn.Module):
    """해상도 보존 ResNet과 TDNN 화자 특징을 함께 사용하는 단일 모델."""

    def __init__(self, pretrained: bool = False, dropout: float = 0.25, spec_augment: bool = True):
        super().__init__()
        self.augment = SpecAugment(max_freq=6, max_time=3) if spec_augment else nn.Identity()
        self.image_backbone = _small_stem_resnet18(pretrained)
        image_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()
        self.audio_encoder = TDNNEncoder(n_mels=64, channels=192, embedding_dim=256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(image_dim + 256),
            nn.Dropout(dropout),
            nn.Linear(image_dim + 256, 2),
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        image_embedding = self.image_backbone(_normalized_rgb(x, self.image_mean, self.image_std))
        audio_embedding = self.audio_encoder(x.squeeze(1))
        return self.classifier(torch.cat([image_embedding, audio_embedding], dim=1))


def build_model(
    pretrained: bool,
    dropout: float = 0.30,
    spec_augment: bool = True,
    model_name: str = "resnet18",
) -> nn.Module:
    if model_name == "resnet18":
        return SpeakerResNet18(pretrained=pretrained, dropout=dropout, spec_augment=spec_augment)
    if model_name == "resnet18_smallstem":
        return SmallStemSpeakerResNet18(
            pretrained=pretrained, dropout=dropout, spec_augment=spec_augment
        )
    if model_name == "hybrid_resnet18_tdnn":
        return HybridSpeakerNet(pretrained=pretrained, dropout=dropout, spec_augment=spec_augment)
    raise ValueError(f"지원하지 않는 model_name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
