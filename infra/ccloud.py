"""Meta-plasticity: Cockroach Cloud cluster scaling via ccloud CLI."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_DENSITY_THRESHOLD = 10_000


@dataclass
class ClusterStatus:
    row_count: int
    threshold: int
    scaling_triggered: bool
    message: str


class CCloudManager:
    """Executes ccloud CLI commands when memory density crosses a threshold."""

    def __init__(
        self,
        cluster_id: str | None = None,
        density_threshold: int = DEFAULT_DENSITY_THRESHOLD,
    ) -> None:
        self.cluster_id = cluster_id or os.getenv("COCKROACH_CLUSTER_ID", "")
        self.density_threshold = density_threshold

    def check_and_scale(self, row_count: int) -> ClusterStatus:
        """Trigger cluster update if semantic memory density exceeds threshold."""
        if row_count < self.density_threshold:
            return ClusterStatus(
                row_count=row_count,
                threshold=self.density_threshold,
                scaling_triggered=False,
                message=f"Density {row_count} below threshold {self.density_threshold}",
            )

        if not self.cluster_id:
            return ClusterStatus(
                row_count=row_count,
                threshold=self.density_threshold,
                scaling_triggered=False,
                message="COCKROACH_CLUSTER_ID not set; skipping scale",
            )

        cmd = [
            "ccloud",
            "cluster",
            "update",
            self.cluster_id,
            "--dedicated-cpu-num",
            "4",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("Meta-plasticity: cluster scaled via ccloud")
                return ClusterStatus(
                    row_count=row_count,
                    threshold=self.density_threshold,
                    scaling_triggered=True,
                    message=result.stdout.strip() or "Cluster update initiated",
                )
            return ClusterStatus(
                row_count=row_count,
                threshold=self.density_threshold,
                scaling_triggered=False,
                message=result.stderr.strip() or "ccloud update failed",
            )
        except FileNotFoundError:
            return ClusterStatus(
                row_count=row_count,
                threshold=self.density_threshold,
                scaling_triggered=False,
                message="ccloud CLI not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return ClusterStatus(
                row_count=row_count,
                threshold=self.density_threshold,
                scaling_triggered=False,
                message="ccloud update timed out",
            )
