# Project Architecture: RouteNet-Fermi Topology Transfer

---

## 1. Project Overview

This project implements RouteNet-Fermi, a graph neural network for predicting delay in computer networks. The core model architecture is implemented in both TensorFlow and PyTorch, with the PyTorch version being the primary implementation for experimentation.

### Core Research Question
**Topology Transfer / Cross-Topology Generalization**: Can a model trained on one network topology (e.g., fat-tree) generalize to unseen topologies (e.g., real ISP networks like Geant, Abilene)?

---

## 2. Directory Structure

```
RouteNet-Fermi-main/
├── topology_transfer/          # PyTorch implementation (primary)
│   ├── routenet_fermi_pytorch.py   # Model architecture
│   ├── train.py                    # Training script (fat128 topology)
│   ├── test.py                     # Flexible evaluation script (AUTO-DISCOVERY)
│   ├── experiments.py               # Experiment & topology discovery engine
│   ├── convert_tf_to_pytorch.py    # TF → PyTorch checkpoint converter
│   ├── data_generator.py           # DatanetAPI → hypergraph converter
│   ├── checkpoints/                 # Saved model checkpoints
│   │   ├── best.pt                 # Best validation MAPE checkpoint
│   │   ├── latest.pt               # Most recent checkpoint
│   │   ├── epoch*.pt               # Per-epoch checkpoints
│   │   ├── converted_from_tf.pt     # Converted from fat_tree TF checkpoint
│   │   └── logs/                   # Training logs
│   └── results/                     # Evaluation results JSON
├── fat_tree/                # Fat-tree topology experiments (TensorFlow original)
│   ├── ckpt_dir_128/        # TF checkpoints for k=128 fat-tree
│   ├── ckpt_dir_64/         # TF checkpoints for k=64 fat-tree
│   └── ckpt_dir_16/         # TF checkpoints for k=16 fat-tree
├── all_mixed/               # Multi-topology training experiment
│   ├── ckpt_dir/            # Full training checkpoints
│   ├── few_shot/            # Few-shot learning checkpoints (5 iters × 8 sample sizes)
│   │   └── ckpt_dir_[0-4]_[25/50/100/500/1000/2000/5000/10000]/
│   └── initial_weights/     # Initial random weights
├── data/                    # Network simulation datasets
│   ├── fat128/              # Fat-tree k=128 topology data
│   │   ├── train/           # Training set (795 samples)
│   │   └── test/            # Test set (198 samples)
│   └── real_traces/         # Real ISP topology data
│       ├── train/           # Training: geant (391 samples)
│       └── test/            # Test: abilene (1000), geant (391), germany50 (288), nobel (71)
├── real_traffic/            # Real traffic data API
│   └── datanetAPI.py        # DatanetAPI wrapper for real topologies
├── datanetAPI.py            # Root-level shared DatanetAPI
├── delay_model.py          # TensorFlow model (reference)
└── ...
```

---

## 3. Model Architecture

RouteNet-Fermi is a graph neural network that operates over a **hypergraph** representation of a network topology:

### 3.1 Hypergraph Nodes
| Node Type | Description | Features |
|-----------|-------------|----------|
| **Path (p)** | End-to-end flow from source to destination | traffic, packets, model, eq_lambda, avg_pkts_lambda, exp_max_factor, pkts_lambda_on, avg_t_off, avg_t_on, ar_a, sigma |
| **Link (l)** | Physical network link/edge | capacity, policy (one-hot: WFQ/SP/DRR/FIFO) |
| **Queue (q)** | Scheduling queue at a router | queue_size, priority (one-hot), weight |

### 3.2 Message Passing (8 iterations)
```
Link/Queue → Path (GRUCell, input=Q+L=64, hidden=32)
Path → Queue (GRUCell, input=P=32, hidden=32)
Queue → Link (GRUCell, input=Q=32, hidden=32)
```

### 3.3 Readout
```
occupancy = MLP(PathState)     → per-hop queueing delay
trans_delay = pkt_size / capacity  → per-hop transmission delay
total_delay = sum(occupancy) + sum(trans_delay)
```

### 3.4 Hyperparameters
| Parameter | Value |
|-----------|-------|
| Path state dim | 32 |
| Link state dim | 32 |
| Queue state dim | 32 |
| Message passing iterations | 8 |
| Max traffic models | 7 |
| Max queues per link | 3 |
| Number of policies | 4 |
| Loss | MAPE (Mean Absolute Percentage Error) |

### 3.5 Z-Score Normalization (hardcoded)
```python
z_score = {
    'traffic': [1385.41, 859.81],
    'packets': [1.40, 0.89],
    'eq_lambda': [1350.97, 858.32],
    'avg_pkts_lambda': [0.91, 0.97],
    'exp_max_factor': [6.66, 4.72],
    'pkts_lambda_on': [0.91, 1.65],
    'avg_t_off': [1.66, 2.36],
    'avg_t_on': [1.66, 2.36],
    'ar_a': [0.0, 1.0],
    'sigma': [0.0, 1.0],
    'capacity': [27611.09, 20090.62],
    'queue_size': [30259.11, 21410.10],
}
```

---

## 4. Available Experiments & Checkpoints

### 4.1 TensorFlow Checkpoints

| Experiment | Location | Epochs | Best Val MAPE | Best Checkpoint |
|---|---|---|---|---|
| Fat-Tree k=128 | `fat_tree/ckpt_dir_128/` | 15 | 0.59% | `02-0.59` |
| Fat-Tree k=64 | `fat_tree/ckpt_dir_64/` | 15 | 0.76% | `15-0.76` |
| Fat-Tree k=16 | `fat_tree/ckpt_dir_16/` | 15 | 0.82% | `01-0.82` |
| All Mixed | `all_mixed/ckpt_dir/` | 20 | 5.44% | `09-5.44` |
| Few-Shot iter=0 | `all_mixed/few_shot/ckpt_dir_0_*/` | 20 | varies | varies |

### 4.2 PyTorch Checkpoints

| File | Source | Val MAPE | Notes |
|---|---|---|---|
| `converted_from_tf.pt` | Converted `fat_tree/ckpt_dir_128/02-0.59` | 0.59% | Ready to use |
| `best.pt` | PyTorch native training | ~19-20% | Needs improvement |
| `epoch01-19.71.pt` | PyTorch native training | 19.71% | First epoch |
| `epoch01-20.27.pt` | PyTorch native training | 20.27% | First epoch |
| `latest.pt` | PyTorch native training | varies | Most recent |

---

## 5. Available Data Topologies

| Name | Kind | Train Samples | Test Samples | Notes |
|---|---|---|---|---|
| `fat128` | fat_tree | 795 | 198 | Fat-tree k=128, matches training |
| `geant` | real_isp | 391 | 391 | European academic network |
| `abilene` | real_isp | 0 | 1000 | US research network |
| `germany50` | real_isp | 0 | 288 | German research network |
| `nobel` | real_isp | 0 | 71 | Nobel network |

**Note**: `all_mixed` dataset is NOT currently downloaded. Download from: https://bnn.upc.edu/download/dataset-v6-all-mixed/

---

## 6. How to Use the Flexible Test System

### 6.1 Interactive Mode (recommended first time)
```bash
cd topology_transfer
python test.py
```
This will show a menu:
```
======================================================================
  Available Checkpoints
======================================================================
  #    Experiment                          Checkpoint              Val MAPE
  ---------------------------------------------------------------------------
  1    Fat-Tree k=128                     epoch02-val_mape0.59    0.59%  [converted]
  2    Fat-Tree k=64                      epoch15-val_mape0.76    0.76%  [converted]
  ...
  [A] Test ALL checkpoints on all available topologies
  [Q] Quit
```
Then select topologies to test.

### 6.2 Command-Line Mode

```bash
# Test specific checkpoint on all topologies
python test.py --ckpt ./checkpoints/converted_from_tf.pt --samples 50

# Test best checkpoint of a specific experiment
python test.py --exp fat_tree_k128 --samples 100

# Test ALL experiments on ALL topologies
python test.py --exp all --topo all --samples 50

# List everything available without running tests
python test.py --list

# Test only specific topologies
python test.py --exp fat_tree_k128 --topo fat128 geant abilene --samples 50
```

### 6.3 Adding New Datasets

To add a new dataset, place it under `data/` with the DatanetAPI structure:

```
data/
├── your_new_topology/
│   └── test/
│       ├── graphs/           # GML topology files
│       ├── routings/         # Routing files
│       └── *.tar.gz          # Simulation result archives
```

The test script will **automatically detect** the new topology on the next run. No code changes needed.

---

## 7. Architecture: Auto-Discovery System

The `experiments.py` module provides a unified discovery engine:

```
experiments.full_scan(project_root)
    ├── scan_topologies()      → discovers all data directories
    ├── scan_checkpoints()     → discovers all TF & PyTorch checkpoints
    └── assign_topologies_to_experiments()
```

Key classes:
- `CheckpointInfo`: Metadata for a single checkpoint (path, epoch, val_mape, source)
- `TopologyInfo`: Metadata for a data topology (name, kind, train/test counts)
- `ExperimentInfo`: Full experiment descriptor with compatible topologies

---

## 8. Key Technical Notes

### 8.1 TF → PyTorch GRU Gate Reordering
TensorFlow and PyTorch use **different gate orderings** in GRUCell:

| Framework | Gate order (columns/rows) |
|-----------|---------------------------|
| TensorFlow | [z=update, r=reset, n=new] |
| PyTorch | [r=reset, z=update, n=new] |

When converting TF checkpoints to PyTorch, the `reorder_gru_weights()` function must reorder the columns/rows accordingly. This has been empirically verified.

### 8.2 Model Parameters
Total parameters: **27,025** (all embeddings + 3 GRUs + readout MLP)
- Path embedding: 17 → 32 (MLP)
- Queue embedding: 5 → 32 (MLP)
- Link embedding: 5 → 32 (MLP)
- Path GRU: (32+32) input, 32 hidden = 4,160 params
- Queue GRU: 32 input, 32 hidden = 3,104 params
- Link GRU: 32 input, 32 hidden = 3,104 params
- Path readout: 32 → 16 → 16 → 1

### 8.3 Training vs Test Data Handling
- **Training**: Uses `fat_tree.datanetAPI.DatanetAPI` with specific `bufferSizes` field
- **Real ISP**: Uses `real_traffic.datanetAPI` with `queueSizes` / `queueSize` fields
- The hypergraph construction automatically detects field availability and computes bandwidth ratios when needed

---

## 9. Training Status

**Known Issue**: PyTorch native training has not yet converged to the expected MAPE (~0.59% on fat128). The converted TF checkpoint achieves 0.59% on fat128 test, but native training shows ~20% MAPE even after multiple epochs.

**Recommended workflow**:
1. Use `converted_from_tf.pt` for cross-topology generalization experiments
2. Fix native training convergence issues separately
3. Few-shot experiments (all_mixed) require downloading the all_mixed dataset

---

## 10. File Change Log

| Date | File | Change |
|------|------|--------|
| 2026-06-02 | `experiments.py` | NEW — auto-discovery engine; per-topology `default_samples` |
| 2026-06-02 | `test.py` | REBUILT — flexible menu + auto-discovery; per-topology sample counts; fixed table KeyError |
| 2026-06-02 | `convert_any.py` | NEW — universal TF→PyTorch converter for all 46 experiments |
| 2026-06-02 | `project architecture.md` | UPDATED — complete rewrite |
