"""Model-update attacks (torch-free)."""

from afkn.attacks.config import ATTACKS, AttackSpec, load_attack_config
from afkn.attacks.plan import AttackPlan, attack_delta

__all__ = ["ATTACKS", "AttackPlan", "AttackSpec", "attack_delta", "load_attack_config"]
