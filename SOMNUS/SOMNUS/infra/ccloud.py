"""ccloud CLI: provisioning and ENVIRONMENT PERTURBATION.

The previous build used ccloud for "meta-plasticity" -- scaling the cluster on
row count. That is autoscaling with a neuroscience label on it: metaplasticity
means a synapse's plasticity depends on its own history, which has nothing to
do with disk pressure. Real metaplasticity now lives in
``sleep_cycle.consolidation.effective_alpha`` as ``alpha * 2^-stability``.

ccloud's honest jobs are provisioning the cluster and perturbing it, which is
what makes it a live demo tool rather than a checkbox.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from infra.config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CCloudResult:
    ok: bool
    command: str
    output: str


class CCloudManager:
    def __init__(self, cluster_id: str | None = None) -> None:
        self.cluster_id = cluster_id or CONFIG.cluster_id

    @staticmethod
    def available() -> bool:
        return shutil.which("ccloud") is not None

    def _run(self, args: list[str], timeout: int = 120) -> CCloudResult:
        if not self.available():
            return CCloudResult(False, " ".join(args), "ccloud CLI not found on PATH")
        try:
            proc = subprocess.run(
                ["ccloud", *args], capture_output=True, text=True, timeout=timeout, check=False
            )
            return CCloudResult(
                proc.returncode == 0, " ".join(args), (proc.stdout or proc.stderr).strip()
            )
        except subprocess.TimeoutExpired:
            return CCloudResult(False, " ".join(args), "timed out")

    def cluster_info(self) -> dict[str, Any]:
        if not self.cluster_id:
            return {"error": "COCKROACH_CLUSTER_ID not set"}
        result = self._run(["cluster", "get", self.cluster_id, "--output", "json"])
        if not result.ok:
            return {"error": result.output}
        try:
            return json.loads(result.output)
        except json.JSONDecodeError:
            return {"raw": result.output}

    def list_nodes(self) -> CCloudResult:
        return self._run(["cluster", "get", self.cluster_id, "--output", "json"])

    def perturb(self, region: str | None = None) -> CCloudResult:
        """Demo perturbation hook.

        Deliberately NOT wired to anything destructive by default -- on stage
        you want a rehearsed, reversible shock, not a surprise. Prefer driving
        the simulator via ``core.control.send_command`` for the live demo and
        keeping real cluster operations manual.
        """
        logger.warning("perturb() is a manual hook; use core.control for demo shocks")
        return CCloudResult(False, "perturb", "not enabled by default -- see docstring")
