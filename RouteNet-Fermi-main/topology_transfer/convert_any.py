"""
Universal TF Checkpoint -> PyTorch Converter
==========================================
Converts ANY RouteNet-Fermi TensorFlow checkpoint directory to PyTorch .pt format.

Reuses the verified conversion logic from convert_tf_to_pytorch.py:
  - reorder_gru_weights() for gate reordering
  - MAPPING for dense layer transforms
  - Shape validation before saving

Supports:
  - fat_tree/ckpt_dir_128/
  - fat_tree/ckpt_dir_64/
  - fat_tree/ckpt_dir_16/
  - all_mixed/ckpt_dir/
  - all_mixed/few_shot/ckpt_dir_*/

Usage:
  python convert_any.py                          # Convert all experiments
  python convert_any.py --exp fat_tree_k128    # Convert specific experiment only
  python convert_any.py --list                  # List available experiments
  python convert_any.py --exp fat_tree_k128 --epoch 2  # Convert specific epoch only
"""

import os
import sys
import re
import glob as glob_mod
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np

# Import the verified conversion logic from the original converter
from topology_transfer.convert_tf_to_pytorch import (
    reorder_gru_weights,
    MAPPING,
    GRU_REORDER_PAIRS,
)


def _load_tf_ckpt_shapes(tf_ckpt_path):
    """Load TF checkpoint variable names and shapes using tensorflow."""
    import tensorflow as tf
    reader = tf.train.load_checkpoint(tf_ckpt_path)
    return reader, reader.get_variable_to_shape_map()


def convert_single_tf_ckpt(tf_ckpt_path, output_path, epoch=0, val_mape=999.0, source_path=None):
    """
    Convert a single TF checkpoint to PyTorch format.
    Validates shapes before saving.
    """
    import tensorflow as tf

    print(f"  Loading: {tf_ckpt_path}")
    reader = tf.train.load_checkpoint(tf_ckpt_path)
    shapes = reader.get_variable_to_shape_map()

    converted = {}

    # --- Dense layers (simple transforms: transpose + copy) ---
    for tf_name, pt_key, transform in MAPPING:
        if transform is not None:
            if tf_name not in shapes:
                print(f"    [SKIP] {tf_name} not found")
                continue
            tf_tensor = reader.get_tensor(tf_name)
            result = transform(tf_tensor)
            converted[pt_key] = torch.from_numpy(result.astype(np.float32))

    # --- GRU layers (reorder gate dimensions) ---
    for gru_name, (kernel_key, rec_key, bias_key) in GRU_REORDER_PAIRS.items():
        if kernel_key not in shapes:
            print(f"    [SKIP] GRU {gru_name}: kernel not found")
            continue
        K = reader.get_tensor(kernel_key)
        R = reader.get_tensor(rec_key)
        B = reader.get_tensor(bias_key)
        print(f"    GRU {gru_name}: K={K.shape}, R={R.shape}, B={B.shape}")

        w_ih, w_hh, b_ih, b_hh = reorder_gru_weights(K, R, B)
        converted[f"{gru_name}.weight_ih"] = torch.from_numpy(w_ih.astype(np.float32).T)
        converted[f"{gru_name}.weight_hh"] = torch.from_numpy(w_hh.astype(np.float32).T)
        converted[f"{gru_name}.bias_ih"] = torch.from_numpy(b_ih.astype(np.float32))
        converted[f"{gru_name}.bias_hh"] = torch.from_numpy(b_hh.astype(np.float32))

    # --- Validate shapes ---
    from topology_transfer.routenet_fermi_pytorch import RouteNetFermiPyTorch
    pt_model = RouteNetFermiPyTorch()
    pt_state = pt_model.state_dict()

    print("  Shape validation:")
    all_ok = True
    for pt_key in sorted(converted.keys()):
        tf_val = converted[pt_key]
        if pt_key in pt_state:
            pt_val = pt_state[pt_key]
            match = "OK" if tf_val.shape == pt_val.shape else f"MISMATCH (PT wants {tuple(pt_val.shape)})"
            if match != "OK":
                all_ok = False
            print(f"    {pt_key}: {tuple(tf_val.shape)}  [{match}]")
        else:
            print(f"    {pt_key}: {tuple(tf_val.shape)}  [NOT FOUND]")
            all_ok = False

    if not all_ok:
        print("  [ERROR] Shape mismatches detected. Skipping this checkpoint.")
        return False

    print(f"  All shapes match! Saving -> {output_path}")

    ckpt = {
        'model_state_dict': converted,
        'epoch': epoch,
        'val_mape': val_mape,
        'source': source_path or tf_ckpt_path,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(ckpt, output_path)
    return True


def parse_tf_checkpoint_dir(ckpt_dir):
    """
    Find all .index files in a TF checkpoint directory.
    Returns list of (epoch, val_mape, tf_ckpt_base_path).
    """
    results = []
    idx_files = glob_mod.glob(os.path.join(ckpt_dir, "*.index"))
    for idx_path in idx_files:
        base = os.path.basename(idx_path)
        m = re.match(r"(\d+)-(\d+\.\d+)\.index", base)
        if m:
            epoch = int(m.group(1))
            mape = float(m.group(2))
            # The TF checkpoint "path" is the .index file without the extension
            tf_base = idx_path.replace(".index", "")
            data_file = tf_base + ".data-00000-of-00001"
            if os.path.exists(data_file):
                results.append((epoch, mape, tf_base))
    results.sort(key=lambda x: (x[1], x[0]))  # sort by mape (best first), then epoch
    return results


def get_output_dir(exp_name):
    """Get the output directory for converted checkpoints."""
    base = os.path.join(project_root, "topology_transfer", "checkpoints", "converted")
    return os.path.join(base, exp_name)


def convert_experiment(ckpt_dir, exp_name, display_name, epoch_filter=None, skip_existing=True, dry_run=False):
    """
    Convert all checkpoints in a TF checkpoint directory.

    Args:
        ckpt_dir: Path to TF checkpoint directory
        exp_name: Experiment identifier (used in output path)
        display_name: Human-readable name
        epoch_filter: If set, only convert this epoch number
        skip_existing: Skip if output .pt file already exists
        dry_run: If True, only print what would be converted
    """
    print(f"\n{'=' * 60}")
    print(f"  {display_name} ({exp_name})")
    print(f"  Source: {ckpt_dir}")
    print(f"{'=' * 60}")

    if not os.path.isdir(ckpt_dir):
        print(f"  [ERROR] Directory not found: {ckpt_dir}")
        return 0, 0

    parsed = parse_tf_checkpoint_dir(ckpt_dir)
    if not parsed:
        print(f"  [WARN] No TF checkpoints found in: {ckpt_dir}")
        return 0, 0

    print(f"  Found {len(parsed)} checkpoint(s):")
    for epoch, mape, tf_base in parsed:
        marker = " *" if epoch == parsed[0][0] else ""
        print(f"    epoch {epoch:02d} | val_mape={mape:.2f}% | {os.path.basename(tf_base)}{marker}")

    out_dir = get_output_dir(exp_name)
    os.makedirs(out_dir, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    for epoch, mape, tf_base in parsed:
        if epoch_filter is not None and epoch != epoch_filter:
            continue

        # Output filename: epochXX-val_mapeY.YY.pt
        out_name = f"epoch{epoch:02d}-val_mape{mape:.2f}.pt"
        out_path = os.path.join(out_dir, out_name)

        if skip_existing and os.path.exists(out_path):
            print(f"\n  [SKIP] Already exists: {out_path}")
            skipped_count += 1
            continue

        if dry_run:
            print(f"\n  [DRY RUN] Would convert: {tf_base} -> {out_path}")
            continue

        print(f"\n  Converting: {os.path.basename(tf_base)} -> {out_name}")
        success = convert_single_tf_ckpt(
            tf_ckpt_path=tf_base,
            output_path=out_path,
            epoch=epoch,
            val_mape=mape,
            source_path=ckpt_dir,
        )
        if success:
            converted_count += 1
        else:
            skipped_count += 1

    return converted_count, skipped_count


# ---------------------------------------------------------------------------
# Known experiment registry
# ---------------------------------------------------------------------------

EXPERIMENT_REGISTRY = {
    "fat_tree_k128": {
        "display": "Fat-Tree k=128",
        "tf_dir": "fat_tree/ckpt_dir_128",
    },
    "fat_tree_k64": {
        "display": "Fat-Tree k=64",
        "tf_dir": "fat_tree/ckpt_dir_64",
    },
    "fat_tree_k16": {
        "display": "Fat-Tree k=16",
        "tf_dir": "fat_tree/ckpt_dir_16",
    },
    "all_mixed": {
        "display": "All Mixed (multi-topology)",
        "tf_dir": "all_mixed/ckpt_dir",
    },
}


def discover_few_shot_experiments():
    """Discover few-shot experiment directories."""
    few_shot_dir = os.path.join(project_root, "all_mixed", "few_shot")
    if not os.path.isdir(few_shot_dir):
        return {}

    experiments = {}
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

        experiments[exp_name] = {
            "display": display,
            "tf_dir": ckpt_sub,
        }
    return experiments


def list_all_experiments():
    """Print all available experiments that can be converted."""
    print("\nAvailable experiments:")
    print(f"  {'Name':<30} {'Display Name':<35} {'Source Dir'}")
    print(f"  {'-' * 30} {'-' * 35} {'-' * 40}")

    all_exp = {}
    for name, info in EXPERIMENT_REGISTRY.items():
        src = os.path.join(project_root, info["tf_dir"])
        status = "READY" if os.path.isdir(src) else "NOT FOUND"
        print(f"  {name:<30} {info['display']:<35} {src} [{status}]")
        all_exp[name] = info

    few_shot = discover_few_shot_experiments()
    if few_shot:
        print()
        print("  Few-shot experiments:")
        for name, info in sorted(few_shot.items()):
            src = info["tf_dir"]
            status = "READY" if os.path.isdir(src) else "NOT FOUND"
            print(f"  {name:<30} {info['display']:<35} [{status}]")
            all_exp[name] = info

    return all_exp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Universal TF -> PyTorch checkpoint converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_any.py --list                    # Show all available experiments
  python convert_any.py --all                    # Convert all experiments
  python convert_any.py --exp fat_tree_k128       # Convert fat_tree k=128 only
  python convert_any.py --exp fat_tree_k64         # Convert fat_tree k=64 only
  python convert_any.py --exp all_mixed           # Convert all_mixed
  python convert_any.py --exp fat_tree_k128 --epoch 2  # Convert specific epoch
  python convert_any.py --all --force             # Re-convert even if .pt exists
        """,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiments that can be converted"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert ALL experiments"
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Experiment name to convert (e.g. fat_tree_k128, all_mixed, few_shot_i0_n100)"
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Only convert this specific epoch number"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .pt files (default: skip them)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be converted without converting"
    )
    args = parser.parse_args()

    print(f"\nProject root: {project_root}")

    # -- List mode
    if args.list:
        all_exp = list_all_experiments()
        out_base = os.path.join(project_root, "topology_transfer", "checkpoints", "converted")
        print(f"\nOutput directory: {out_base}")
        print(f"\nExisting converted checkpoints:")
        if os.path.isdir(out_base):
            for exp_dir in sorted(os.listdir(out_base)):
                exp_path = os.path.join(out_base, exp_dir)
                if os.path.isdir(exp_path):
                    files = [f for f in os.listdir(exp_path) if f.endswith(".pt")]
                    if files:
                        print(f"  {exp_dir}: {', '.join(sorted(files))}")
                    else:
                        print(f"  {exp_dir}: (empty)")
        return

    # Build the list of experiments to convert
    to_convert = []

    if args.exp:
        # Specific experiment
        exp_name = args.exp

        # Check registry
        if exp_name in EXPERIMENT_REGISTRY:
            info = EXPERIMENT_REGISTRY[exp_name]
            to_convert.append((exp_name, info["display"], os.path.join(project_root, info["tf_dir"])))
        # Check few-shot
        elif exp_name.startswith("few_shot_"):
            few_shot = discover_few_shot_experiments()
            if exp_name in few_shot:
                info = few_shot[exp_name]
                to_convert.append((exp_name, info["display"], info["tf_dir"]))
            else:
                print(f"[ERROR] Unknown experiment: {exp_name}")
                print("Run with --list to see available experiments.")
                return
        else:
            print(f"[ERROR] Unknown experiment: {exp_name}")
            print("Run with --list to see available experiments.")
            return

    elif args.all:
        # All experiments from registry
        for exp_name, info in EXPERIMENT_REGISTRY.items():
            src = os.path.join(project_root, info["tf_dir"])
            to_convert.append((exp_name, info["display"], src))

        # All few-shot experiments
        few_shot = discover_few_shot_experiments()
        for exp_name, info in sorted(few_shot.items()):
            # Skip if already covered
            if not any(name == exp_name for name, _, _ in to_convert):
                to_convert.append((exp_name, info["display"], info["tf_dir"]))

    else:
        print("Nothing to do. Use --list, --all, or --exp.")
        print("Run with --list to see available experiments.")
        return

    # -- Run conversions --
    print(f"\n{'=' * 60}")
    print(f"  Converting {len(to_convert)} experiment(s)")
    print(f"{'=' * 60}")

    total_converted = 0
    total_skipped = 0

    for exp_name, display, tf_dir in to_convert:
        converted, skipped = convert_experiment(
            ckpt_dir=tf_dir,
            exp_name=exp_name,
            display_name=display,
            epoch_filter=args.epoch,
            skip_existing=not args.force,
            dry_run=args.dry_run,
        )
        total_converted += converted
        total_skipped += skipped

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"  Converted: {total_converted} | Skipped (exists): {total_skipped}")
    out_base = os.path.join(project_root, "topology_transfer", "checkpoints", "converted")
    print(f"  Output: {out_base}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
