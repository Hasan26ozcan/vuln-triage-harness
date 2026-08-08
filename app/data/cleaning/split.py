"""Stage 2, step 1 + step 5: repo-based split with an immutable, seeded
manifest.

Hard constraint (this is the whole point of the stage): every sample from
the same `repo_name` lands in the same split. Splitting a repo across
train/test would let the model see one commit from a repo during training
and be "tested" on a different commit from the *same* codebase — that's
leakage, and it's exactly what this project's interview story hinges on
having prevented (see README's evaluation-metrics honesty section).

Class balance (step 4) is a *soft* objective layered on top of that hard
constraint via greedy bin-packing: whole repo-groups (never split) are
assigned one at a time, largest group first, to whichever split currently
(a) is furthest under its target size, with a bonus for (b) not yet having
any samples of this group's CWE classes. This is a heuristic, not an
optimal solver — with real CVEfixes data (mostly one CVE per repo) it
converges close to the target ratios in practice; `check_class_balance`
in balance.py is what actually verifies the outcome, not this function's
promise.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.schemas.vuln import VulnSample

SPLIT_RATIOS: dict[str, float] = {"train": 0.7, "val": 0.1, "test": 0.1, "gold_eval": 0.1}

# How strongly "fills a CWE class this split doesn't have yet" outweighs
# raw size deficit when picking a split for a repo-group. Tuned informally
# by hand, not learned — revisit if check_class_balance keeps flagging gaps.
_CLASS_COVERAGE_BONUS = 5.0


@dataclass
class SplitManifest:
    seed: int
    ratios: dict[str, float]
    assignment: dict[str, str]  # repo_name -> split

    def to_json(self) -> dict:
        return {"seed": self.seed, "ratios": self.ratios, "assignment": self.assignment}

    @classmethod
    def from_json(cls, data: dict) -> SplitManifest:
        return cls(seed=data["seed"], ratios=data["ratios"], assignment=data["assignment"])


def _group_by_repo(samples: list[VulnSample]) -> dict[str, list[VulnSample]]:
    groups: dict[str, list[VulnSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.repo_name].append(sample)
    return groups


def leakage_safe_split(
    samples: list[VulnSample],
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> tuple[list[VulnSample], SplitManifest]:
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios.values())}")
    if not samples:
        return [], SplitManifest(seed=seed, ratios=dict(ratios), assignment={})

    groups = _group_by_repo(samples)
    total = len(samples)
    targets = {name: ratio * total for name, ratio in ratios.items()}
    current_counts = {name: 0 for name in ratios}
    current_class_counts: dict[str, Counter] = {name: Counter() for name in ratios}

    # Largest group first: placing big groups earlier leaves more room to
    # correct for their size when placing the smaller ones later. Shuffle
    # first, with a seeded PRNG, so equal-size groups don't always land the
    # same way — this is deterministic tie-breaking for reproducibility,
    # not a security context, so the stdlib PRNG is the right tool here.
    rng = random.Random(seed)  # nosec B311
    repo_names = list(groups.keys())
    rng.shuffle(repo_names)
    repo_names.sort(key=lambda name: len(groups[name]), reverse=True)

    assignment: dict[str, str] = {}
    for repo_name in repo_names:
        group = groups[repo_name]
        group_cwe_counts = Counter(s.cwe_id for s in group)

        best_split, best_score = None, None
        for split_name in ratios:
            size_deficit = targets[split_name] - current_counts[split_name]
            new_classes_covered = sum(
                1 for cwe_id in group_cwe_counts if current_class_counts[split_name][cwe_id] == 0
            )
            score = size_deficit + _CLASS_COVERAGE_BONUS * new_classes_covered
            if best_score is None or score > best_score:
                best_score, best_split = score, split_name

        assignment[repo_name] = best_split
        current_counts[best_split] += len(group)
        current_class_counts[best_split].update(group_cwe_counts)

    updated_samples = [s.model_copy(update={"split": assignment[s.repo_name]}) for s in samples]
    manifest = SplitManifest(seed=seed, ratios=dict(ratios), assignment=assignment)
    return updated_samples, manifest


def apply_manifest(samples: list[VulnSample], manifest: SplitManifest) -> list[VulnSample]:
    """Re-apply a previously saved split assignment instead of recomputing
    it. This is what makes the split actually immutable across pipeline
    re-runs (step 5) — once a manifest is saved, later runs call this,
    not leakage_safe_split() again, so experiment results stay comparable.
    """
    updated = []
    for sample in samples:
        split = manifest.assignment.get(sample.repo_name)
        if split is None:
            raise ValueError(
                f"No split assignment for repo {sample.repo_name!r} in this manifest "
                f"(new repo added after the manifest was frozen?)."
            )
        updated.append(sample.model_copy(update={"split": split}))
    return updated


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json(), indent=2))


def load_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.from_json(json.loads(Path(path).read_text()))
