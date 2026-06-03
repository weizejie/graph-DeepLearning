"""
Experiment & Topology Discovery Engine
=====================================
Automatically scans the project for:
1. All available checkpoints (TF .index files and PyTorch .pt files)
2. All available data topologies (tar.gz archives grouped by topology type)
3. The mapping between experiments and which topologies they can be tested on

This module is the single source of truth for all discovery logic.
New datasets / checkpoints added to the project are automatically picked up.
"""

import os
import re
import glob as glob_mod
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint."""
    path: str           # Absolute path to the checkpoint file
    experiment: str      # Human-readable experiment name, e.g. "fat_tree_k128"
    source: str         # "tf" or "pt"
    epoch: int           # Epoch number
    val_mape: float     # Validation MAPE from filename (TF) or metadata (PyTorch)
    ckpt_name: str      # Short name e.g. "epoch02-val_mape0.59"

    @property
    def label(self) -> str:
        tag = "converted" if self.source == "tf" else "native_pt"
        return f"{self.ckpt_name} ({tag})"


@dataclass
class TopologyInfo:
    """Metadata for a data topology directory."""
    name: str           # Short name e.g. "abilene", "geant", "fat128"
    kind: str            # "fat_tree" | "real_isp" | "all_mixed"
    data_dir: str        # Absolute path to the topology's data dir
    has_train: bool      # Has train/ subdirectory
    has_test: bool       # Has test/ subdirectory
    num_train: int = 0   # Number of .tar.gz files in train/
    num_test: int = 0    # Number of .tar.gz files in test/
    api_class_hint: str = "real"  # "fat128" | "real" | "all_mixed"
    default_samples: int = 50  # Default max_samples for testing this topology


@dataclass
class ExperimentInfo:
    """Full experiment descriptor."""
    name: str            # Unique experiment identifier e.g. "fat_tree_k128"
    display_name: str    # Human-readable e.g. "Fat-Tree k=128"
    tf_ckpt_dir: Optional[str] = None  # Path to TF checkpoint directory
    pt_ckpts: list = field(default_factory=list)  # List of CheckpointInfo (PyTorch)
    data_topologies: list = field(default_factory=list)  # List of TopologyInfo
    description: str = ""
    # Which topology kinds this experiment can be tested on:
    compatible_kinds: list = field(default_factory=lambda: ["fat_tree"])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_tar_files(top_dir: str) -> int:
    if not os.path.isdir(top_dir):
        return 0
    return len(glob_mod.glob(os.path.join(top_dir, "*.tar.gz")))


def _parse_tf_checkpoint(ckpt_dir: str):
    """
    Parse all .index files in a TF checkpoint directory.
    Returns list of (epoch, val_mape, file_path).
    """
    results = []
    idx_files = glob_mod.glob(os.path.join(ckpt_dir, "*.index"))
    for idx_path in idx_files:
        base = os.path.basename(idx_path)
        m = re.match(r"(\d+)-(\d+\.\d+)\.index", base)
        if m:
            epoch = int(m.group(1))
            mape = float(m.group(2))
            data_file = idx_path.replace(".index", ".data-00000-of-00001")
            if os.path.exists(data_file):
                results.append((epoch, mape, idx_path.replace(".index", "")))
    results.sort(key=lambda x: (x[1], x[0]))  # sort by mape then epoch
    return results


def _detect_kind_from_path(path: str) -> str:
    """
    Detect the kind of data based on the directory path.
    """
    path_lower = path.lower()
    if "fat128" in path_lower or "fat_tree" in path_lower or "fattree" in path_lower:
        return "fat_tree"
    elif "all_mixed" in path_lower or "mixed" in path_lower:
        return "all_mixed"
    else:
        return "real_isp"


def _detect_api_class(path: str) -> str:
    path_lower = path.lower()
    if "fat128" in path_lower:
        return "fat128"
    elif "all_mixed" in path_lower:
        return "all_mixed"
    else:
        return "real"


def _get_topo_name_from_dir(topo_dir: str) -> str:
    """Extract a clean topology name from its directory."""
    return os.path.basename(topo_dir)


# ---------------------------------------------------------------------------
# Scanning functions
# ---------------------------------------------------------------------------

def scan_topologies(project_root: str) -> list[TopologyInfo]:
    """
    Scan the entire project for all available data topologies.
    Walks data/ and any other directories containing topology data.

    Looks for:
        data/fat128/test/  (fat128 tar.gz files)
        data/real_traces/test/{abilene,geant,...}/  (real ISP tar.gz files)
        all_mixed/data/{train,test}/  (all_mixed tar.gz files)

    Returns:
        List of TopologyInfo sorted by name.
    """
    topologies = []
    data_root = os.path.join(project_root, "data")
    all_mixed_data = os.path.join(project_root, "all_mixed", "data")

    # ---- fat128 ----
    fat128_test = os.path.join(project_root, "data", "fat128", "test")
    if os.path.isdir(fat128_test):
        num_test = _count_tar_files(fat128_test)
        if num_test > 0:
                topologies.append(TopologyInfo(
                name="fat128",
                kind="fat_tree",
                data_dir=fat128_test,
                has_train=os.path.isdir(os.path.join(project_root, "data", "fat128", "train")),
                has_test=True,
                num_test=num_test,
                num_train=_count_tar_files(os.path.join(project_root, "data", "fat128", "train")),
                api_class_hint="fat128",
                default_samples=10,  # fat128 has 198 test files, 10 is enough for quick test
            ))

    # ---- real ISP traces ----
    real_root = os.path.join(project_root, "data", "real_traces")
    if os.path.isdir(real_root):
        for split in ("test", "train"):
            split_dir = os.path.join(real_root, split)
            if not os.path.isdir(split_dir):
                continue
            for topo_name in os.listdir(split_dir):
                topo_path = os.path.join(split_dir, topo_name)
                if not os.path.isdir(topo_path):
                    continue
                # Check for graphs subdir (DatanetAPI convention)
                if not os.path.exists(os.path.join(topo_path, "graphs")):
                    continue
                num = _count_tar_files(topo_path)
                if num == 0:
                    continue
                # Avoid duplicates (merge train + test under same name)
                existing = next((t for t in topologies if t.name == topo_name), None)
                if existing:
                    if split == "train":
                        existing.has_train = True
                        existing.num_train = num
                    else:
                        existing.has_test = True
                        existing.num_test = num
                else:
                    topologies.append(TopologyInfo(
                        name=topo_name,
                        kind="real_isp",
                        data_dir=topo_path,
                        has_train=(split == "train"),
                        has_test=(split == "test"),
                        num_train=(num if split == "train" else 0),
                        num_test=(num if split == "test" else 0),
                        api_class_hint="real",
                    ))

    # ---- all_mixed ----
    for split in ("train", "test"):
        split_dir = os.path.join(all_mixed_data, split)
        if not os.path.isdir(split_dir):
            continue
        topo_name = "all_mixed"
        num = _count_tar_files(split_dir)
        if num == 0:
            continue
        existing = next((t for t in topologies if t.name == topo_name), None)
        if existing:
            if split == "train":
                existing.has_train = True
                existing.num_train = num
            else:
                existing.has_test = True
                existing.num_test = num
        else:
            topologies.append(TopologyInfo(
                name=topo_name,
                kind="all_mixed",
                data_dir=split_dir,
                has_train=(split == "train"),
                has_test=(split == "test"),
                num_train=(num if split == "train" else 0),
                num_test=(num if split == "test" else 0),
                api_class_hint="all_mixed",
            ))

    topologies.sort(key=lambda t: t.name)
    return topologies


def scan_checkpoints(project_root: str) -> list[ExperimentInfo]:
    """
    Scan all experiment directories and collect checkpoints.

    Returns:
        List of ExperimentInfo sorted by name.
    """
    experiments = {}
    pt_ckpts_dir = os.path.join(project_root, "topology_transfer", "checkpoints")

    # ---- fat_tree experiments ----
    fat_tree_dir = os.path.join(project_root, "fat_tree")
    if os.path.isdir(fat_tree_dir):
        for sub_dir in os.listdir(fat_tree_dir):
            ckpt_sub = os.path.join(fat_tree_dir, sub_dir)
            if not os.path.isdir(ckpt_sub):
                continue
            idx_files = glob_mod.glob(os.path.join(ckpt_sub, "*.index"))
            if not idx_files:
                continue

            # Parse experiment name, e.g. "ckpt_dir_128" -> "fat_tree_k128"
            m = re.match(r"ckpt_dir_(\d+)", sub_dir)
            if m:
                k = m.group(1)
                exp_name = f"fat_tree_k{k}"
            else:
                exp_name = f"fat_tree_{sub_dir}"
            display = f"Fat-Tree k={k}" if m else f"Fat-Tree {sub_dir}"

            # Parse TF checkpoints
            tf_parsed = _parse_tf_checkpoint(ckpt_sub)
            best_tf = tf_parsed[0] if tf_parsed else None  # best = lowest mape

            key = exp_name
            experiments[key] = ExperimentInfo(
                name=exp_name,
                display_name=display,
                tf_ckpt_dir=ckpt_sub,
                pt_ckpts=[],
                description=f"Fat-Tree topology with k={k}" if m else sub_dir,
                compatible_kinds=["fat_tree", "real_isp"],
            )

    # ---- all_mixed experiment ----
    all_mixed_dir = os.path.join(project_root, "all_mixed")
    if os.path.isdir(all_mixed_dir):
        ckpt_dir = os.path.join(all_mixed_dir, "ckpt_dir")
        if os.path.isdir(ckpt_dir):
            tf_parsed = _parse_tf_checkpoint(ckpt_dir)
            if tf_parsed:
                experiments["all_mixed"] = ExperimentInfo(
                    name="all_mixed",
                    display_name="All Mixed (multi-topology)",
                    tf_ckpt_dir=ckpt_dir,
                    pt_ckpts=[],
                    description="Trained on mixed topologies for generalization",
                    compatible_kinds=["fat_tree", "real_isp", "all_mixed"],
                )

        # ---- few-shot experiments ----
        few_shot_dir = os.path.join(all_mixed_dir, "few_shot")
        if os.path.isdir(few_shot_dir):
            for sub_dir in os.listdir(few_shot_dir):
                ckpt_sub = os.path.join(few_shot_dir, sub_dir)
                if not os.path.isdir(ckpt_sub):
                    continue
                idx_files = glob_mod.glob(os.path.join(ckpt_sub, "*.index"))
                if not idx_files:
                    continue

                m = re.match(r"ckpt_dir_(\d+)_(\d+)", sub_dir)
                if m:
                    iter_idx = m.group(1)
                    n_samples = m.group(2)
                    exp_name = f"few_shot_i{iter_idx}_n{n_samples}"
                    display = f"Few-Shot iter={iter_idx}, n={n_samples}"
                else:
                    exp_name = f"few_shot_{sub_dir}"
                    display = f"Few-Shot {sub_dir}"

                experiments[exp_name] = ExperimentInfo(
                    name=exp_name,
                    display_name=display,
                    tf_ckpt_dir=ckpt_sub,
                    pt_ckpts=[],
                    description=display,
                    compatible_kinds=["fat_tree", "real_isp", "all_mixed"],
                )

    # ---- Converted checkpoints (from convert_any.py) ----
    # Path: checkpoints/converted/<exp_name>/epochXX-val_mapeY.Y.pt
    converted_dir = os.path.join(pt_ckpts_dir, "converted")
    if os.path.isdir(converted_dir):
        for exp_sub in os.listdir(converted_dir):
            exp_dir = os.path.join(converted_dir, exp_sub)
            if not os.path.isdir(exp_dir):
                continue
            for f in os.listdir(exp_dir):
                if not f.endswith(".pt"):
                    continue
                pt_path = os.path.join(exp_dir, f)

                # Infer display name from directory
                display_map = {
                    "fat_tree_k128": "Fat-Tree k=128 (converted)",
                    "fat_tree_k64": "Fat-Tree k=64 (converted)",
                    "fat_tree_k16": "Fat-Tree k=16 (converted)",
                    "all_mixed": "All Mixed (converted)",
                }
                display = display_map.get(exp_sub, f"{exp_sub} (converted)")

                # Parse epoch and mape from filename / metadata
                epoch = 0
                val_mape = 999.0
                epoch_m = re.search(r"epoch(\d+)", f)
                mape_m = re.search(r"(\d+\.\d+)", f)
                if epoch_m:
                    epoch = int(epoch_m.group(1))
                if mape_m:
                    val_mape = float(mape_m.group(1))

                try:
                    import torch as _torch
                    ckpt = _torch.load(pt_path, map_location="cpu", weights_only=True)
                    if isinstance(ckpt, dict):
                        epoch = ckpt.get("epoch", epoch)
                        vm = ckpt.get("val_mape", None)
                        if vm is not None:
                            val_mape = float(vm)
                except Exception:
                    pass

                stem = os.path.splitext(f)[0]
                if epoch > 0 or val_mape < 999.0:
                    ckpt_name = f"epoch{epoch:02d}-val_mape{val_mape:.2f}"
                else:
                    ckpt_name = stem

                c = CheckpointInfo(
                    path=pt_path,
                    experiment=exp_sub,
                    source="tf",
                    epoch=epoch,
                    val_mape=val_mape,
                    ckpt_name=ckpt_name,
                )

                if exp_sub not in experiments:
                    experiments[exp_sub] = ExperimentInfo(
                        name=exp_sub,
                        display_name=display,
                        pt_ckpts=[],
                        description=display,
                        compatible_kinds=["fat_tree", "real_isp"],
                    )
                experiments[exp_sub].pt_ckpts.append(c)

    # ---- PyTorch native checkpoints (checkpoints/*.pt, NOT in converted/) ----
    if os.path.isdir(pt_ckpts_dir):
        for f in os.listdir(pt_ckpts_dir):
            if not f.endswith(".pt"):
                continue
            pt_path = os.path.join(pt_ckpts_dir, f)
            # Skip files that are actually in the converted directory
            if "converted" in f.lower():
                continue

            # Determine experiment label
            if "epoch" in f.lower() or "best" in f.lower() or "latest" in f.lower():
                exp_name = "pytorch_native"
                display = "PyTorch Native"
                source = "pt"
            else:
                exp_name = "pytorch_native"
                display = "PyTorch Native"
                source = "pt"

            # Parse epoch and mape from filename
            epoch = 0
            val_mape = 999.0
            epoch_m = re.search(r"epoch(\d+)", f)
            mape_m = re.search(r"(\d+\.\d+)", f)
            if epoch_m:
                epoch = int(epoch_m.group(1))
            if mape_m:
                val_mape = float(mape_m.group(1))

            # For native PyTorch, epoch from checkpoint metadata is more reliable
            try:
                import torch as _torch
                ckpt = _torch.load(pt_path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict):
                    epoch = ckpt.get("epoch", epoch)
                    vm = ckpt.get("val_mape", None)
                    if vm is not None:
                        val_mape = float(vm)
            except Exception:
                pass

            stem = os.path.splitext(f)[0]
            if epoch > 0 or val_mape < 999.0:
                ckpt_name = f"epoch{epoch:02d}-val_mape{val_mape:.2f} [{stem}]"
            else:
                ckpt_name = f"{stem}"

            c = CheckpointInfo(
                path=pt_path,
                experiment=exp_name,
                source=source,
                epoch=epoch,
                val_mape=val_mape,
                ckpt_name=ckpt_name,
            )

            if exp_name not in experiments:
                experiments[exp_name] = ExperimentInfo(
                    name=exp_name,
                    display_name=display,
                    pt_ckpts=[],
                    description=display,
                )
            experiments[exp_name].pt_ckpts.append(c)

    # Sort pt_ckpts by val_mape
    for exp in experiments.values():
        exp.pt_ckpts.sort(key=lambda c: c.val_mape)

    # Ensure converted experiments and pytorch_native can test on all available topologies
    for exp in experiments.values():
        if (exp.tf_ckpt_dir is not None
            or "converted" in exp.display_name.lower()
            or exp.name.startswith("fat_tree")
            or exp.name.startswith("all_mixed")):
            exp.compatible_kinds = ["fat_tree", "real_isp"]

    result = sorted(experiments.values(), key=lambda e: e.name)
    return result


def assign_topologies_to_experiments(
    experiments: list[ExperimentInfo],
    topologies: list[TopologyInfo],
) -> list[ExperimentInfo]:
    """
    For each experiment, assign the compatible topologies based on
    the experiment's compatible_kinds.
    """
    for exp in experiments:
        exp.data_topologies = [
            t for t in topologies
            if t.kind in exp.compatible_kinds
            or (t.kind == "fat_tree" and "fat_tree" in exp.compatible_kinds)
            or (t.kind == "real_isp" and "real_isp" in exp.compatible_kinds)
            or (t.kind == "all_mixed" and "all_mixed" in exp.compatible_kinds)
        ]
        # For fat_tree experiments, always include fat128 + real ISP
        if exp.name.startswith("fat_tree"):
            exp.data_topologies = [
                t for t in topologies
                if t.kind in ("fat_tree", "real_isp")
            ]
    return experiments


# ---------------------------------------------------------------------------
# Convenience queries
# ---------------------------------------------------------------------------

def get_experiment_by_name(experiments: list[ExperimentInfo], name: str) -> Optional[ExperimentInfo]:
    return next((e for e in experiments if e.name == name), None)


def get_topology_by_name(topologies: list[TopologyInfo], name: str) -> Optional[TopologyInfo]:
    return next((t for t in topologies if t.name == name), None)


def get_best_checkpoint(experiment: ExperimentInfo) -> Optional[CheckpointInfo]:
    """Return the best (lowest MAPE) PyTorch checkpoint for an experiment."""
    if not experiment.pt_ckpts:
        return None
    return experiment.pt_ckpts[0]


# ---------------------------------------------------------------------------
# Summary / print helpers
# ---------------------------------------------------------------------------

def print_topology_summary(topologies: list[TopologyInfo]):
    print("\n[Topology Discovery Results]")
    print(f"  {'Name':<15} {'Kind':<15} {'Train':<12} {'Test':<12} {'API Hint':<12}")
    print(f"  {'-'*15} {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    for t in topologies:
        train_str = f"{t.num_train}" if t.has_train else "N/A"
        test_str = f"{t.num_test}" if t.has_test else "N/A"
        print(f"  {t.name:<15} {t.kind:<15} {train_str:<12} {test_str:<12} {t.api_class_hint:<12}")


def print_experiment_summary(experiments: list[ExperimentInfo], topologies: list[TopologyInfo]):
    print("\n[Experiment Discovery Results]")
    print(f"  Found {len(experiments)} experiments with {sum(len(e.pt_ckpts) for e in experiments)} PyTorch checkpoints")
    print()
    for i, exp in enumerate(experiments):
        print(f"  [{i+1}] {exp.display_name} ({exp.name})")
        print(f"      Description: {exp.description}")
        if exp.tf_ckpt_dir:
            tf_parsed = _parse_tf_checkpoint(exp.tf_ckpt_dir)
            if tf_parsed:
                best = tf_parsed[0]
                print(f"      TF best: epoch={best[0]}, val_mape={best[1]:.2f}% -> {os.path.basename(best[2])}")
            else:
                print(f"      TF dir: {exp.tf_ckpt_dir} (no .index files found)")
        if exp.pt_ckpts:
            best = exp.pt_ckpts[0]
            print(f"      PT best: {best.ckpt_name} ({best.path})")
        else:
            print(f"      PT checkpoints: None")
        topo_names = [t.name for t in exp.data_topologies]
        print(f"      Compatible topologies: {', '.join(topo_names) if topo_names else 'None'}")
        print()


# ---------------------------------------------------------------------------
# Main scan (one-call discovery)
# ---------------------------------------------------------------------------

def full_scan(project_root: str) -> tuple[list[ExperimentInfo], list[TopologyInfo]]:
    """
    Perform a full project scan and return (experiments, topologies).
    Call this once at startup.
    """
    topologies = scan_topologies(project_root)
    experiments = scan_checkpoints(project_root)
    experiments = assign_topologies_to_experiments(experiments, topologies)
    return experiments, topologies
