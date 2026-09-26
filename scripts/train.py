"""Train the HumanDiT^DH teacher.

uv run python scripts/train.py configs/laptop_smoke.yaml
python scripts/train.py configs/pod_teacher_b.yaml budget.budget_usd=4.0 budget.usd_per_hour=0.34
"""

from __future__ import annotations

import sys

from omegaconf import OmegaConf

from humangen.train.trainer import Trainer


def main() -> None:
    usage = (
        __doc__ + "\nusage: train.py CONFIG.yaml [key=value ...]  (dotlist overrides on configs/base.yaml)"
    )
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(usage)
        return
    if len(sys.argv) < 2:
        sys.exit(usage)
    cfg_path, *overrides = sys.argv[1:]
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"), OmegaConf.load(cfg_path), OmegaConf.from_dotlist(overrides)
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
