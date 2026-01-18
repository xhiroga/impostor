from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .base import EvaluationModel, MetricResult


class SSIMModel(EvaluationModel):
    """Structural Similarity (SSIM) metric against a reference frame."""

    name = "ssim"

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        if window_size % 2 == 0:
            raise ValueError("SSIM window_size must be odd.")
        self.window_size = window_size
        self.sigma = sigma
        self.base_window = self._build_window(window_size, sigma)

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        if len(frame_indices) < 2:
            raise ValueError("SSIM evaluation needs at least two frames.")

        reference = frames[frame_indices[0]].unsqueeze(0)
        results: list[MetricResult] = []
        ssim_values: list[float] = []

        for idx in frame_indices[1:]:
            target = frames[idx].unsqueeze(0)
            score = float(self._ssim(reference, target).item())
            ssim_values.append(score)
            results.append(
                MetricResult(
                    model=self.name,
                    metric="ssim_pair",
                    value=score,
                    frames_used=(frame_indices[0], idx),
                    detail=f"ref={frame_indices[0]}, cmp={idx}",
                )
            )

        mean_score = sum(ssim_values) / len(ssim_values)
        results.append(
            MetricResult(
                model=self.name,
                metric="ssim_mean",
                value=mean_score,
                frames_used=tuple(frame_indices),
                detail="mean of all pairwise SSIM scores against reference frame 0",
            )
        )
        return results

    def _build_window(self, window_size: int, sigma: float) -> torch.Tensor:
        # Construct a 2D Gaussian kernel normalized to sum to 1.
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel_2d = (gauss[:, None] @ gauss[None, :]).unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # img1/img2: shape (1, C, H, W)
        channel = img1.size(1)
        device = img1.device
        dtype = img1.dtype

        window = self.base_window.to(device=device, dtype=dtype).expand(
            channel, 1, self.window_size, self.window_size
        )

        padding = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean()
