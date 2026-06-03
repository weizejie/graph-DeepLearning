"""
Flexible Test Script for RouteNet-Fermi (PyTorch)
==================================================
Auto-discovers all available checkpoints and datasets, then lets you
select which ones to evaluate via an interactive menu.

Usage:
    python test.py                      # Interactive mode (menu-driven)
    python test.py --ckpt path/to/x.pt  # Use specific checkpoint
    python test.py --all               # Test best checkpoint on all topologies
    python test.py --exp fat_tree_k128 # Test best checkpoint of an experiment
    python test.py --exp all --samples 50 # Test all experiments on all topologies
    python test.py --list             # Just list what's available
"""

import os
import sys
import re
import glob as glob_mod
import argparse
import json
import traceback
from typing import Optional

import numpy as np
import torch
import networkx as nx

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "real_traffic"))

from topology_transfer.routenet_fermi_pytorch import RouteNetFermiPyTorch
from topology_transfer import experiments

from real_traffic.datanetAPI import DatanetAPI as DatanetAPI_real
from datanetAPI import DatanetAPI as DatanetAPI_fat128

POLICIES = np.array(["WFQ", "SP", "DRR", "FIFO"])

DATASET_API_MAP = {
    "fat128": DatanetAPI_fat128,
    "real": DatanetAPI_real,
    "all_mixed": DatanetAPI_real,
}


# ---------------------------------------------------------------------------
# Hypergraph conversion (shared with train.py)
# ---------------------------------------------------------------------------

def network_to_hypergraph(G, R, T, P, has_flow_level_delay=True):
    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if G.has_edge(src, dst):
                    policy_str = G.nodes[src]["schedulingPolicy"] if "schedulingPolicy" in G.nodes[src] else "FIFO"
                    policy_val = np.where(policy_str == POLICIES)[0]
                    policy_code = policy_val[0] if len(policy_val) > 0 else 3
                    D_G.add_node(
                        "l_{}_{}".format(src, dst),
                        capacity=G[src][dst][0]["bandwidth"] if G.is_multigraph() else G.edges[src, dst]["bandwidth"],
                        policy=policy_code,
                    )
                agg_delay = P[src, dst]["AggInfo"]["AvgDelay"]
                for f_id in range(len(T[src, dst]["Flows"])):
                    if T[src, dst]["Flows"][f_id]["AvgBw"] != 0 and T[src, dst]["Flows"][f_id]["PktsGen"] != 0:
                        time_dist_params = [0] * 8
                        flow = T[src, dst]["Flows"][f_id]
                        model = 0
                        flow["TimeDistParams"]["Distribution"] = "Poisson"
                        if "EqLambda" in flow["TimeDistParams"]:
                            time_dist_params[0] = flow["TimeDistParams"]["EqLambda"]
                        if "AvgPktsLambda" in flow["TimeDistParams"]:
                            time_dist_params[1] = flow["TimeDistParams"]["AvgPktsLambda"]
                        if "ExpMaxFactor" in flow["TimeDistParams"]:
                            time_dist_params[2] = flow["TimeDistParams"]["ExpMaxFactor"]
                        if "PktsLambdaOn" in flow["TimeDistParams"]:
                            time_dist_params[3] = flow["TimeDistParams"]["PktsLambdaOn"]
                        if "AvgTOff" in flow["TimeDistParams"]:
                            time_dist_params[4] = flow["TimeDistParams"]["AvgTOff"]
                        if "AvgTOn" in flow["TimeDistParams"]:
                            time_dist_params[5] = flow["TimeDistParams"]["AvgTOn"]
                        if "AR-a" in flow["TimeDistParams"]:
                            time_dist_params[6] = flow["TimeDistParams"]["AR-a"]
                        if "sigma" in flow["TimeDistParams"]:
                            time_dist_params[7] = flow["TimeDistParams"]["sigma"]

                        flow_delay = (
                            P[src, dst]["Flows"][f_id]["AvgDelay"]
                            if has_flow_level_delay and P[src, dst]["Flows"][f_id]["AvgDelay"] > 0
                            else agg_delay
                        )

                        D_G.add_node(
                            "p_{}_{}_{}".format(src, dst, f_id),
                            source=src, destination=dst,
                            tos=int(T[src, dst]["Flows"][0]["ToS"]),
                            traffic=T[src, dst]["Flows"][f_id]["AvgBw"],
                            packets=T[src, dst]["Flows"][f_id]["PktsGen"],
                            length=len(R[src, dst]) - 1,
                            model=model,
                            eq_lambda=time_dist_params[0],
                            avg_pkts_lambda=time_dist_params[1],
                            exp_max_factor=time_dist_params[2],
                            pkts_lambda_on=time_dist_params[3],
                            avg_t_off=time_dist_params[4],
                            avg_t_on=time_dist_params[5],
                            ar_a=time_dist_params[6],
                            sigma=time_dist_params[7],
                            delay=flow_delay,
                        )

                        for h_1, h_2 in [R[src, dst][i : i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                            D_G.add_edge("l_{}_{}".format(h_1, h_2), "p_{}_{}_{}".format(src, dst, f_id))
                            D_G.add_edge("p_{}_{}_{}".format(src, dst, f_id), "l_{}_{}".format(h_1, h_2))
                            if "bufferSizes" in G.nodes[h_1]:
                                q_s = str(G.nodes[h_1]["bufferSizes"]).split(",")
                            elif "queueSizes" in G.nodes[h_1]:
                                bw_ratio = T[src, dst]["Flows"][f_id]["AvgBw"] / T[src, dst]["Flows"][f_id]["PktsGen"]
                                q_s = [str(int(q) * bw_ratio) for q in str(G.nodes[h_1]["queueSizes"]).split(",")]
                            elif "queueSize" in G.nodes[h_1]:
                                bw_ratio = T[src, dst]["Flows"][f_id]["AvgBw"] / T[src, dst]["Flows"][f_id]["PktsGen"]
                                q_s = [str(int(q) * bw_ratio) for q in str(G.nodes[h_1]["queueSize"]).split(",")]
                            else:
                                q_s = ["1000"]
                            if "schedulingWeights" in G.nodes[h_1]:
                                if G.nodes[h_1]["schedulingWeights"] != "-":
                                    q_w = [float(w) for w in str(G.nodes[h_1]["schedulingWeights"]).split(",")]
                                    q_w = [w / sum(q_w) for w in q_w]
                                else:
                                    q_w = ["-"]
                            else:
                                q_w = ["-"]
                            if "tosToQoSqueue" in G.nodes[h_1]:
                                q_map = [m.split(",") for m in str(G.nodes[h_1]["tosToQoSqueue"]).split(";")]
                            else:
                                q_map = [["0"], ["1"], ["2"]]
                            q_n = 0
                            n_queues = G.nodes[h_1]["levelsQoS"] if "levelsQoS" in G.nodes[h_1] else 1
                            for q in range(n_queues):
                                D_G.add_node(
                                    "q_{}_{}_{}".format(h_1, h_2, q),
                                    queue_size=int(q_s[q]) if q < len(q_s) else 1000,
                                    priority=q_n,
                                    weight=q_w[q] if q_w[0] != "-" else 0,
                                )
                                D_G.add_edge("q_{}_{}_{}".format(h_1, h_2, q), "l_{}_{}".format(h_1, h_2))
                                if str(int(T[src, dst]["Flows"][0]["ToS"])) in q_map[q]:
                                    D_G.add_edge("p_{}_{}_{}".format(src, dst, f_id), "q_{}_{}_{}".format(h_1, h_2, q))
                                    D_G.add_edge("q_{}_{}_{}".format(h_1, h_2, q), "p_{}_{}_{}".format(src, dst, f_id))
                                q_n += 1

    D_G.remove_nodes_from([node for node, in_degree in D_G.in_degree() if in_degree == 0])
    return D_G


def hypergraph_to_input_data(HG):
    n_q, n_p, n_l = 0, 0, 0
    mapping = {}
    for entity in list(HG.nodes()):
        if entity.startswith("q"):
            mapping[entity] = "q_{}".format(n_q)
            n_q += 1
        elif entity.startswith("p"):
            mapping[entity] = "p_{}".format(n_p)
            n_p += 1
        elif entity.startswith("l"):
            mapping[entity] = "l_{}".format(n_l)
            n_l += 1
    HG = nx.relabel_nodes(HG, mapping)

    link_to_path, queue_to_path, path_to_queue = [], [], []
    queue_to_link, path_to_link = [], []

    for node in HG.nodes:
        in_nodes = [s for s, d in HG.in_edges(node)]
        if node.startswith("q_"):
            path = []
            for n in in_nodes:
                if n.startswith("p_"):
                    path_pos = [d for _, d in HG.out_edges(n) if d.startswith("q_")]
                    path.append([int(n.replace("p_", "")), path_pos.index(node) if node in path_pos else 0])
            path_to_queue.append(path)
        elif node.startswith("p_"):
            links, queues = [], []
            for n in in_nodes:
                if n.startswith("l_"):
                    links.append(int(n.replace("l_", "")))
                elif n.startswith("q_"):
                    queues.append(int(n.replace("q_", "")))
            link_to_path.append(links)
            queue_to_path.append(queues)
        elif node.startswith("l_"):
            queues, paths = [], []
            for n in in_nodes:
                if n.startswith("q_"):
                    queues.append(int(n.replace("q_", "")))
                elif n.startswith("p_"):
                    path_pos = [d for _, d in HG.out_edges(n) if d.startswith("l_")]
                    paths.append([int(n.replace("p_", "")), path_pos.index(node) if node in path_pos else 0])
            path_to_link.append(paths)
            queue_to_link.append(queues)

    attrs = lambda HG, key: list(nx.get_node_attributes(HG, key).values())

    return {
        "traffic": np.expand_dims(attrs(HG, "traffic"), 1),
        "packets": np.expand_dims(attrs(HG, "packets"), 1),
        "length": attrs(HG, "length"),
        "model": attrs(HG, "model"),
        "eq_lambda": np.expand_dims(attrs(HG, "eq_lambda"), 1),
        "avg_pkts_lambda": np.expand_dims(attrs(HG, "avg_pkts_lambda"), 1),
        "exp_max_factor": np.expand_dims(attrs(HG, "exp_max_factor"), 1),
        "pkts_lambda_on": np.expand_dims(attrs(HG, "pkts_lambda_on"), 1),
        "avg_t_off": np.expand_dims(attrs(HG, "avg_t_off"), 1),
        "avg_t_on": np.expand_dims(attrs(HG, "avg_t_on"), 1),
        "ar_a": np.expand_dims(attrs(HG, "ar_a"), 1),
        "sigma": np.expand_dims(attrs(HG, "sigma"), 1),
        "capacity": np.expand_dims(attrs(HG, "capacity"), 1),
        "queue_size": np.expand_dims(attrs(HG, "queue_size"), 1),
        "policy": attrs(HG, "policy"),
        "priority": attrs(HG, "priority"),
        "weight": np.expand_dims(attrs(HG, "weight"), 1),
        "delay": attrs(HG, "delay"),
        "link_to_path": link_to_path,
        "queue_to_path": queue_to_path,
        "queue_to_link": queue_to_link,
        "path_to_queue": path_to_queue,
        "path_to_link": path_to_link,
    }


def prepare_inputs(data, device="cpu"):
    num_paths = len(data["traffic"])
    num_links = len(data["capacity"])
    num_queues = len(data["queue_size"])

    return {
        "traffic": torch.tensor(data["traffic"], dtype=torch.float32, device=device),
        "packets": torch.tensor(data["packets"], dtype=torch.float32, device=device),
        "length": torch.tensor(data["length"], dtype=torch.long, device=device),
        "model": torch.tensor(data["model"], dtype=torch.long, device=device),
        "eq_lambda": torch.tensor(data["eq_lambda"], dtype=torch.float32, device=device),
        "avg_pkts_lambda": torch.tensor(data["avg_pkts_lambda"], dtype=torch.float32, device=device),
        "exp_max_factor": torch.tensor(data["exp_max_factor"], dtype=torch.float32, device=device),
        "pkts_lambda_on": torch.tensor(data["pkts_lambda_on"], dtype=torch.float32, device=device),
        "avg_t_off": torch.tensor(data["avg_t_off"], dtype=torch.float32, device=device),
        "avg_t_on": torch.tensor(data["avg_t_on"], dtype=torch.float32, device=device),
        "ar_a": torch.tensor(data["ar_a"], dtype=torch.float32, device=device),
        "sigma": torch.tensor(data["sigma"], dtype=torch.float32, device=device),
        "capacity": torch.tensor(data["capacity"], dtype=torch.float32, device=device),
        "policy": torch.tensor(data["policy"], dtype=torch.long, device=device),
        "queue_size": torch.tensor(data["queue_size"], dtype=torch.float32, device=device),
        "priority": torch.tensor(data["priority"], dtype=torch.long, device=device),
        "weight": torch.tensor(data["weight"], dtype=torch.float32, device=device),
        "link_to_path": data["link_to_path"],
        "queue_to_path": data["queue_to_path"],
        "path_to_link": data["path_to_link"],
        "path_to_queue": data["path_to_queue"],
        "queue_to_link": data["queue_to_link"],
        "path_lens": torch.tensor(data["length"], dtype=torch.long, device=device),
        "num_paths": num_paths,
        "num_queues": num_queues,
        "num_links": num_links,
    }


def compute_mape(pred, target):
    mask = target > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100)


def evaluate_topology(model, topo_info, DatanetAPI_cls, device, max_samples, has_flow_level_delay):
    """
    Evaluate a model on a single topology.
    topo_info: experiments.TopologyInfo
    Returns: (mape, mae, rmse, n_samples) or (None, None, None, 0) on error.
    """
    model.eval()
    all_preds, all_targets = [], []
    sample_count = 0

    try:
        tool = DatanetAPI_cls(topo_info.data_dir, shuffle=False)
    except Exception as e:
        print(f"    [ERROR] Cannot open {topo_info.data_dir}: {e}")
        return None, None, None, 0

    tar_files = glob_mod.glob(os.path.join(topo_info.data_dir, "*.tar.gz"))
    total_tar = len(tar_files) if tar_files else max_samples
    desc = topo_info.name

    for sample in tool:
        if sample_count >= max_samples:
            break

        label = f"[{desc} {sample_count + 1}/{min(max_samples, total_tar)}]"

        try:
            G = nx.DiGraph(sample.get_topology_object())
            T = sample.get_traffic_matrix()
            R = sample.get_routing_matrix()
            P = sample.get_performance_matrix()
        except Exception as e:
            continue

        try:
            HG = network_to_hypergraph(G=G, R=R, T=T, P=P, has_flow_level_delay=has_flow_level_delay)
            data = hypergraph_to_input_data(HG)
        except Exception as e:
            continue

        delay = np.array(data.pop("delay"), dtype=np.float32)
        if len(delay) == 0 or not all(x > 0 for x in delay):
            continue

        try:
            inputs = prepare_inputs(data, device)
        except Exception as e:
            continue

        try:
            with torch.no_grad():
                pred = model(inputs)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(delay)
            sample_count += 1
        except Exception as e:
            continue

        print(f"\r    {label} paths={len(data['traffic'])}, links={len(data['capacity'])}, queues={len(data['queue_size'])}    ", end="", flush=True)

    print(flush=True)

    if not all_preds:
        return None, None, None, 0

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mape = compute_mape(all_preds, all_targets)
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    return mape, mae, rmse, len(all_preds)


def load_checkpoint(model, ckpt_path, device):
    """Load a PyTorch checkpoint into the model. Returns val_mape from metadata."""
    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        val_mape = ckpt.get("val_mape", None)
        return val_mape
    except Exception as e:
        print(f"  [ERROR] Failed to load checkpoint: {e}")
        return None


def print_separator(char="=", width=70):
    print(char * width)


def print_section(title):
    print()
    print_separator("=")
    print("  " + title)
    print_separator("=")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu(experiments_list, topologies_list):
    """
    Show an interactive menu and return the list of (checkpoint_path, experiment, topology) tuples to test.
    """
    valid_experiments = [e for e in experiments_list if e.pt_ckpts]
    if not valid_experiments:
        print("[ERROR] No valid checkpoints found. Use --list to see what's available.")
        return None

    print_section("Available Checkpoints")
    print(f"  {'#':<4} {'Experiment':<35} {'Checkpoint':<25} {'Val MAPE':<10}")
    print("  " + "-" * 78)
    choice_map = {}
    idx = 1
    for exp in valid_experiments:
        for ckpt in exp.pt_ckpts[:3]:  # Show at most 3 per experiment
            tag = "converted" if ckpt.source == "tf" else "native_pt"
            print(f"  {idx:<4} {exp.display_name:<35} {ckpt.ckpt_name:<25} {ckpt.val_mape:>6.2f}%  [{tag}]")
            choice_map[idx] = (ckpt.path, exp, ckpt)
            idx += 1

    print()
    print("  [A] Test ALL checkpoints above on all available topologies")
    print("  [Q] Quit")
    print()

    raw = input("Enter choice(s) [e.g. 1, 3-5, or A]: ").strip().lower()

    if raw == "q":
        return None

    selected = []

    if raw == "a":
        # All checkpoints
        selected = list(choice_map.values())
    else:
        # Parse individual and range selections
        parts = raw.replace(",", " ").split()
        for part in parts:
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    for i in range(int(start), int(end) + 1):
                        if i in choice_map:
                            selected.append(choice_map[i])
                except ValueError:
                    pass
            else:
                try:
                    i = int(part)
                    if i in choice_map:
                        selected.append(choice_map[i])
                except ValueError:
                    pass

    if not selected:
        print("[WARN] No valid choices. Nothing to test.")
        return None

    # Deduplicate by checkpoint path
    seen = set()
    deduped = []
    for item in selected:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    selected = deduped

    # Ask about topology scope
    print_section("Available Topologies to Test")
    print(f"  {'#':<4} {'Name':<15} {'Kind':<15} {'Test files':<12} {'Default samples':<18}")
    print("  " + "-" * 70)
    topo_map = {}
    for j, topo in enumerate(topologies_list):
        if topo.has_test and topo.num_test > 0:
            print(f"  {j+1:<4} {topo.name:<15} {topo.kind:<15} {topo.num_test:<12} (default: {topo.default_samples})")
            topo_map[j + 1] = topo

    print()
    print("  [A] Test on ALL available topologies")
    print("  [Q] Go back")
    raw_t = input("Test on which topologies [e.g. 1, 3-5, or A]: ").strip().lower()

    if raw_t == "q":
        return interactive_menu(experiments_list, topologies_list)

    selected_topos = []
    if raw_t == "a":
        selected_topos = list(topo_map.values())
    else:
        parts_t = raw_t.replace(",", " ").split()
        for part in parts_t:
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    for i in range(int(start), int(end) + 1):
                        if i in topo_map:
                            selected_topos.append(topo_map[i])
                except ValueError:
                    pass
            else:
                try:
                    i = int(part)
                    if i in topo_map:
                        selected_topos.append(topo_map[i])
                except ValueError:
                    pass

    if not selected_topos:
        print("[WARN] No topologies selected.")
        return None

    return [(ckpt_path, exp, topo) for (ckpt_path, exp, ckpt) in selected for topo in selected_topos]

# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_tests(selected_items, max_samples, device, output_path):
    """
    selected_items: list of (ckpt_path, experiment, topo_info)
    Runs all evaluations and prints results.
    """
    results = {}  # {ckpt_path: {topo_name: {mape, mae, rmse, n}}}

    # Group items by checkpoint to minimize model reloads
    ckpt_items = {}  # {ckpt_path: (exp, [(topo, samples), ...])}
    for ckpt_path, exp, topo_info in selected_items:
        # Per-topology sample count: CLI override (cap) or topology default
        n_samples = min(topo_info.default_samples, max_samples)
        if ckpt_path not in ckpt_items:
            ckpt_items[ckpt_path] = (exp, [])
        ckpt_items[ckpt_path][1].append((topo_info, n_samples))

    for ckpt_path, (exp, topo_list) in ckpt_items.items():
        print(f"\n  Loading checkpoint: {ckpt_path}")
        model = RouteNetFermiPyTorch().to(device)
        val_mape = load_checkpoint(model, ckpt_path, device)
        print(f"  Experiment: {exp.display_name} | Val MAPE from training: {val_mape:.2f}%" if val_mape else "  [no val_mape in checkpoint]")

        if ckpt_path not in results:
            results[ckpt_path] = {"experiment": exp.display_name, "ckpt_name": os.path.basename(ckpt_path), "topologies": {}}

        for topo_info, n_samples in topo_list:
            api_cls = DATASET_API_MAP.get(topo_info.api_class_hint, DatanetAPI_real)
            has_flow_level = topo_info.kind in ("all_mixed",)

            print(f"\n  Testing on: {topo_info.name} (max {n_samples} samples, from {topo_info.data_dir})")
            mape, mae, rmse, n = evaluate_topology(model, topo_info, api_cls, device, n_samples, has_flow_level)

            if mape is not None:
                results[ckpt_path]["topologies"][topo_info.name] = {
                    "mape": mape,
                    "mae": mae,
                    "rmse": rmse,
                    "n_samples": n,
                    "max_requested": n_samples,
                }
                print(f"  => MAPE={mape:.2f}% | MAE={mae:.4f} | RMSE={rmse:.4f} | N={n}")
            else:
                results[ckpt_path]["topologies"][topo_info.name] = {"mape": None, "error": "no valid samples"}
                print(f"  => SKIPPED (no valid samples)")

    return results


def print_results_table(results):
    """Print a clean results summary table."""
    if not results:
        return

    print_separator("=")
    print("  RESULTS SUMMARY")
    print_separator("=")

    # Gather all topology names (skip __config__)
    all_topos = set()
    for k, r in results.items():
        if k == "__config__":
            continue
        all_topos.update(r["topologies"].keys())
    all_topos = sorted(all_topos)

    # Column widths (skip __config__)
    ckpt_col = max((len(r["ckpt_name"]) for r in results.values() if "ckpt_name" in r), default=20) + 2
    exp_col = max((len(r["experiment"]) for r in results.values() if "experiment" in r), default=20) + 2
    topo_col = max(max(len(t) for t in all_topos), 12) + 2

    header = f"  {'Checkpoint':<{ckpt_col}} {'Experiment':<{exp_col}}"
    for topo in all_topos:
        header += f" {topo:<{topo_col}}"
    print(header)
    sep = "  " + "-" * (ckpt_col + exp_col) + "+" + ("-" * (topo_col + 2) * len(all_topos))
    print(sep)

    for k, r in results.items():
        if k == "__config__":
            continue
        if "topologies" not in r:
            continue
        row = f"  {r['ckpt_name']:<{ckpt_col}} {r['experiment']:<{exp_col}}"
        for topo in all_topos:
            if topo in r["topologies"]:
                entry = r["topologies"][topo]
                if entry["mape"] is not None:
                    row += f" {entry['mape']:>{topo_col - 1}.2f}%"
                else:
                    row += f" {'N/A':<{topo_col}}"
            else:
                row += f" {'--':<{topo_col}}"
        print(row)

    print()
    print("  Note: MAPE = Mean Absolute Percentage Error (lower is better)")
    print(f"  All tests used max_samples={results.get('__config__', {}).get('max_samples', '?')}")


def save_results(results, output_path):
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flexible RouteNet-Fermi evaluator: auto-discovers checkpoints & datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py                        # Interactive menu
  python test.py --list                 # List all available experiments & topologies
  python test.py --ckpt ./checkpoints/best.pt  # Test specific checkpoint
  python test.py --all                  # Test best checkpoint on all topologies
  python test.py --exp fat_tree_k128    # Test best checkpoint of an experiment
  python test.py --exp all --samples 100 # Test all experiments, 100 samples each
  python test.py --topo fat128 abilene  # Test only selected topologies
  python test.py --convert-all          # Convert all TF checkpoints to PyTorch
        """,
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to a specific PyTorch checkpoint (.pt file)"
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Experiment name (e.g. fat_tree_k128, all_mixed). 'all' = test all."
    )
    parser.add_argument(
        "--topo", type=str, nargs="+", default=None,
        help="Topology name(s) to test (e.g. fat128 abilene geant). 'all' = all."
    )
    parser.add_argument(
        "--samples", "--max_samples", dest="max_samples", type=int, default=50,
        help="Max samples per topology (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="./topology_transfer/results/test_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )
    parser.add_argument(
        "--list", dest="list_only", action="store_true",
        help="Only list available experiments and topologies, don't run tests"
    )
    parser.add_argument(
        "--convert-all", dest="convert_all", action="store_true",
        help="Convert all TF checkpoints to PyTorch format first"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"\nDevice: {device}")

    # ---- Full project scan ----
    print("Scanning project for experiments and topologies...")
    experiments_list, topologies_list = experiments.full_scan(project_root)

    if args.list_only:
        experiments.print_experiment_summary(experiments_list, topologies_list)
        experiments.print_topology_summary(topologies_list)
        return

    # ---- List-only if nothing found ----
    if not experiments_list and not topologies_list:
        print("[ERROR] No experiments or topologies found.")
        print("  - Did you download the datasets?")
        print("  - Check that checkpoints exist in fat_tree/ and all_mixed/")
        experiments.print_topology_summary(topologies_list)
        return

    # ---- Build the test plan ----
    selected_items = []  # list of (ckpt_path, experiment, topo_info)

    # -- Mode 1: Specific checkpoint --
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            print(f"[ERROR] Checkpoint not found: {args.ckpt}")
            return
        # Find matching experiment
        matched_exp = None
        for exp in experiments_list:
            for ckpt in exp.pt_ckpts:
                if os.path.normpath(ckpt.path) == os.path.normpath(args.ckpt):
                    matched_exp = exp
                    break
        if matched_exp is None:
            matched_exp = experiments.ExperimentInfo(
                name="custom",
                display_name=os.path.basename(args.ckpt),
                pt_ckpts=[],
            )
        # Topologies
        if args.topo and args.topo != ["all"]:
            selected_topos = [t for t in topologies_list if t.name in args.topo and t.has_test]
        else:
            selected_topos = [t for t in topologies_list if t.has_test]
        selected_items = [(args.ckpt, matched_exp, t) for t in selected_topos]

    # -- Mode 2: Specific experiment --
    elif args.exp:
        if args.exp == "all":
            target_exps = experiments_list
        else:
            target_exps = [e for e in experiments_list if e.name == args.exp]
            if not target_exps:
                print(f"[ERROR] Unknown experiment: {args.exp}")
                print("Available experiments:")
                for e in experiments_list:
                    print(f"  - {e.name} ({e.display_name})")
                return

        # Topologies
        if args.topo and args.topo != ["all"]:
            selected_topos = [t for t in topologies_list if t.name in args.topo and t.has_test]
        else:
            selected_topos = [t for t in topologies_list if t.has_test]

        for exp in target_exps:
            if not exp.pt_ckpts:
                print(f"[WARN] No checkpoints for experiment: {exp.name}")
                continue
            best_ckpt = exp.pt_ckpts[0]  # lowest mape
            for topo in selected_topos:
                selected_items.append((best_ckpt.path, exp, topo))

    # -- Mode 3: Interactive menu --
    else:
        items = interactive_menu(experiments_list, topologies_list)
        if items is None:
            print("Exiting.")
            return
        selected_items = items

    if not selected_items:
        print("[ERROR] Nothing selected to test. Run with --list to see what's available.")
        return

    # Deduplicate
    seen = set()
    deduped = []
    for item in selected_items:
        key = (item[0], item[2].name)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    selected_items = deduped

    print(f"\nTest plan: {len(selected_items)} evaluation(s)")

    # ---- Convert TF checkpoints if needed ----
    if args.convert_all:
        print_section("Converting TF Checkpoints")
        from topology_transfer import convert_tf_to_pytorch as converter
        for ckpt_path, exp, _ in selected_items:
            # Check if this is a TF checkpoint that hasn't been converted
            if not os.path.exists(ckpt_path):
                tf_dir = exp.tf_ckpt_dir
                if tf_dir:
                    print(f"  Converting {exp.display_name}...")
                    try:
                        converter.convert(
                            tf_dir,
                            os.path.join(project_root, "topology_transfer", "checkpoints",
                                        f"converted_{exp.name}.pt")
                        )
                    except Exception as e:
                        print(f"  [ERROR] Conversion failed: {e}")
            # Reload experiments to get new checkpoints
        experiments_list, topologies_list = experiments.full_scan(project_root)

    # ---- Run evaluations ----
    print_section("Running Evaluations")
    results = run_tests(selected_items, args.max_samples, device, args.output)

    # ---- Print summary ----
    results["__config__"] = {"max_samples": args.max_samples, "device": str(device)}
    print_results_table(results)

    # ---- Save ----
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
