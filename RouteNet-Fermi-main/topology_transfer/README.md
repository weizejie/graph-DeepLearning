# RouteNet-Fermi PyTorch - Topology Transfer Experiment

## Overview

PyTorch reimplementation of RouteNet-Fermi to avoid TensorFlow version conflicts.
Evaluates the model's cross-topology generalization ability.

**Goal**: Train on fat128 topology (128-node fat-tree), test on real ISP traffic (GEANT, Abilene, Nobel).

## Files

| File | Description |
|------|-------------|
| `routenet_fermi_pytorch.py` | PyTorch model |
| `data_generator.py` | Hypergraph construction & data loading |
| `train.py` | Training on fat128 |
| `test.py` | Cross-topology evaluation |
| `run_experiment.bat` | Full pipeline |
| `requirements.txt` | Dependencies |
| `README.md` | This file |

## Quick Start

### 1. Install dependencies

```bash
pip install torch networkx numpy
```

### 2. Train on fat128

```bash
cd topology_transfer
python train.py --train_dir ../data/fat128/train --val_dir ../data/fat128/test --epochs 10
```

### 3. Test (fat128 + real simultaneously)

```bash
python test.py --all
```

This tests **fat128** (same-topology) alongside all real topologies (geant, abilene, nobel, germany50) and prints them side-by-side for easy comparison.

To test a subset only:

```bash
python test.py --dataset fat128          # fat128 only
python test.py --dataset real           # real traffic only
```

Or run everything at once:
```bash
run_experiment.bat
```

## Output

| Directory | Content |
|-----------|---------|
| `checkpoints/best.pt` | Best model weights |
| `checkpoints/latest.pt` | Most recent checkpoint |
| `results/` | Test evaluation results |

## Experiment Design

- **Training**: fat128 (128-node fat-tree datacenter topology)
- **Testing**: Real ISP topologies (GEANT=22 nodes, Abilene=12, Nobel=38)
- **Metric**: MAPE (Mean Absolute Percentage Error)

## Model Architecture

RouteNet-Fermi uses 8 iterations of GNN-style message passing over a hypergraph:

1. **Path GRU**: Aggregates queue + link states per path hop, step-by-step
2. **Queue GRU**: Sums path states at queue positions
3. **Link GRU**: Aggregates queue states

Final delay = queueing delay + transmission delay per path.
