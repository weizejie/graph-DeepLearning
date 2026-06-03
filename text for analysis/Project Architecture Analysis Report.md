# Graph DeepLearning Project Architecture Analysis Report

> Generated: 2026-05-28
> Based on: execution.md, The background analysis.md, GraphiT paper reading report.md, Topics to learn.md, report/ directory, RouteNet-Fermi-main/ source code, T3former code/ source code

---

## 1. Project Overview

This is a **graph deep learning research project in computer networking**, comprising two implemented lines of work and one exploratory fusion direction.

### Three Technical Lines

**Line A: RouteNet-Fermi (TensorFlow)**
- Task: Network performance prediction (Delay / Jitter / Loss)
- Method: Heterogeneous graph message passing (Path/Link/Queue nodes + GRU)
- Status: Fully implemented

**Line B: T3Former (PyTorch + PyG)**
- Task: Temporal graph classification
- Method: Transformer x 2 + GraphSAGE + Attention Fusion
- Features: Sliding window DoS + Betti numbers
- Status: Fully implemented

**Line C: Fusion Exploration (New Model, Low Documentation)**
- Task: Same as RouteNet (network performance prediction)
- Method: Inherit RouteNet's 3-node hypergraph + Replace MPNN with Transformer
- Features: Flow Token as [CLS] + RoPE + Local GNN + Bias-free FFN + GeGLU
- Status: Described only in execution.md, code not yet implemented

---

## 2. Implemented Modules

### 2.1 RouteNet-Fermi (Implemented)

#### Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | TensorFlow / Keras |
| Graph API | datanetAPI (custom network topology interface) |
| Datasets | fat128 (Fat-Tree 128 synthetic), real_traces (GEANT, Abilene, Germany50, Nobel) |

#### Core Architecture

**Data Flow:**
1. Input: network topology (GML) + traffic matrix + routing matrix + performance matrix
2. `network_to_hypergraph()` builds hypergraph
3. Hypergraph contains Path/Link/Queue three node types
4. Embedding layers: three independent MLPs (Path=17-dim, Queue=5-dim, Link=5-dim -> all 32-dim)
5. Message passing: 8 iterations, bidirectional GRU Message Passing
6. Readout layer: Dense(16) -> Dense(16) -> Dense(1)

**Key Parameters:**
- Iterations: 8 (hardcoded)
- Hidden dimensions: path=32, link=32, queue=32
- Normalization (z_score): traffic, packets, eq_lambda, avg_pkts_lambda, exp_max_factor, pkts_lambda_on, avg_t_off, avg_t_on, ar_a, sigma, capacity, queue_size

#### Experiment Branches (Implemented)

| Directory | Content |
|-----------|---------|
| traffic_models/ | delay/jitter/loss prediction under different traffic models |
| scheduling/ | performance prediction under different queue scheduling policies |
| scalability/ | network scale experiments |
| all_mixed/ | mixed scenarios, few-shot / full training |
| real_traffic/ | real traffic data validation |
| testbed/ | testbed experiments |
| fat_tree/ | Fat-Tree topology experiments |
| exercise/ | 4-level learning exercises |

**Open Questions:**
- What is the difference between all_mixed/ and real_traffic/? Do they fine-tune on the same model architecture?
- What is the few-shot definition? How large is the gap vs full training?

---

### 2.2 T3Former (Implemented)

#### Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch + PyTorch Geometric |
| Graph ops | GCNConv, SAGEConv, GATConv, GINConv, TransformerConv, GPSConv, Graphormer, etc. |
| Topological features | pyflagser (persistent homology / Betti numbers), DoS (Degree of Spectrality) histogram |

#### Core Architecture

**Three-branch fusion:**
1. Branch 1: TransformerClassifier (temporal feature 1, sliding window feature X0)
2. Branch 2: TransformerClassifier (temporal feature 2, sliding window feature X1)
3. Branch 3: GraphSAGE (graph structure feature, PyG Data format)
4. Attention Fusion: Multi-Head Attention fuses three branch outputs
5. FC: classification output

**Feature extraction:** Sliding window (window=6, stride=4) extracts Betti-0, Betti-1, DoS histogram

#### Datasets

| Dataset | Type | Task |
|---------|------|------|
| pemsbay, pems04, pems08 | Traffic networks | Binary classification |
| mit_ct1, highschool_ct1, facebook_ct1, dblp_ct1, tumblr_ct1, infectious_ct1 | Social networks | Binary classification |

#### Other Implemented Baseline Models

T3former code/ contains: TGN, TGAT (Temporal Graph Attention Networks), GraphMixer, GCN_LSTM, EvolveGCN, and more temporal graph model implementations.

**Open Questions:**
- Is the sliding window parameter (window=6, stride=4) fixed or tunable? Does optimal window size vary across datasets?
- The DoS + Betti feature calculation details need further documentation
- T3Former was originally for graph classification: is there intent to migrate to network performance prediction?

---

## 3. New Fusion Model: Transformer + RouteNet Architecture

### 3.1 Core Design (from execution.md)

This is an exploratory direction that replaces RouteNet's MPNN message passing with a Transformer while keeping the 3-node hypergraph.

**Key Changes:**

| Component | RouteNet-Fermi (existing) | New Fusion (exploratory) |
|-----------|---------------------------|--------------------------|
| Framework | TensorFlow | PyTorch (likely) |
| Core method | MPNN (GRU, 8 iterations) | Transformer (global Attention) |
| Node types | Path/Link/Queue | Inherited |
| Perception range | Single-hop local | Flow Token global |
| Topology encoding | None | RoPE + Local GNN |
| FFN | Dense(RELU) | Bias-free Dense + GeGLU |
| Loss | MSE | Log-MSE |

### 3.2 Tokenizer Design (Most Open Questions)

**Design Assumption:** One flow = one attention sliding window

```
[FLOW_TOKEN] [HOP_TOKEN_q1,l1] [HOP_TOKEN_q2,l2] ... [HOP_TOKEN_qn,ln]
  ([CLS])         (hop 1)           (hop 2)              (hop n)

Hop Token composition: [Queue embedding, Link embedding] via some operation
```

**Open Questions:**

Q1: HOP_TOKEN dimension - Queue(32-dim) + Link(32-dim) = concatenation (64-dim)? Addition (32-dim)? Gating (32-dim)?

Q2: Variable path lengths - What is the max hop count? How is padding handled? What is the attention mask design?

Q3: FLOW_TOKEN position - At the beginning of all hop tokens (BERT [CLS] style)? As the sole query for each flow? Participates in cross-attention?

Q4: Attention scope - Within same flow: flow token attends all hop tokens, hop tokens attend each other? Cross-flow: do tokens from different flows attend each other? How are cross-graph tokens handled?

Q5: FLOW_TOKEN input - Flow feature embedding? Raw flow features? Aggregated info from all queue/link along the path?

Q6: Precise meaning of "flow as an attention sliding window" - Inspired by Transformer-XL's sliding window attention? Or just that token sequence length equals hop count?

Q7: Multi-timestep data handling - Each timestep modeled as a separate flow sequence? Concatenated into one long sequence? Per-timestep independent batch?

### 3.3 Topology Embedding (Two Components)

#### 3.3.1 RoPE Relative Position Encoding

Q8: Definition of "position" - hop index (hop 1, hop 2...)? Graph-theoretic distance? Topological coordinates in the original graph?

每个hop作为一个position， 位置只算跳数差

Q9: RoPE frequency selection - Standard RoPE uses multiple sin/cos frequencies. How is theta chosen? Does max path length determine max position index?

遵循经典rope

Q10: Relative vs absolute position - RoPE encodes absolute position, achieving relative position awareness via rotation. Is additional relative position bias (ALiBi) needed?

#### 3.3.2 Local GNN Neighborhood Integration

Q11: Definition of "neighborhood" - Topological: adjacent link/queue in the original graph? Sequential: adjacent hop tokens on the path? Both?

这个邻居正是依照原图的邻域

Q12: GNN architecture choice - GCN / GAT / GraphSAGE? Why this one over others?

学习t3former，使用graphsage

Q13: "Every x nodes as one learning group" - x = how many? Fixed or learnable? Aggregation within group? Is this a graph coarsening strategy?

Q14: Position of local GNN - Preprocessing before Transformer? Integrated into each Transformer layer? Concatenated with RoPE?

### 3.4 Encoder Details

Q15: FFN expansion ratio 2.66 - 32 x 2.66 = 85.12 -> 85. Source of this ratio? Theoretical or empirical basis?

源自geglu门控因为多引入一个矩阵，而拓展4倍维度倍视为有效的，所以4*2/3=2.66

Q16: Encoder layer count - How many Transformer layers? Corresponding to RouteNet's 8 iterations?

先尝试8层

Q17: Attention head count - How many heads? Dimension per head?

Q18: Layer Normalization - Pre-LN / Post-LN / RMSNorm? Major impact on training stability.

Q19: Motivation for bias-free FFN - Match RoPE's unbiased design? Or other regularization considerations?

Q20: Loss function log-MSE - log(pred + eps) - log(true + eps) squared? Or log(MSE)? Impact on large vs small values?

---

## 4. Relationship Between Two Implemented Modules

RouteNet-Fermi (network performance prediction, TensorFlow, heterogeneous graph MPNN) and T3Former (temporal graph classification, PyTorch + PyG, static graph + temporal features) have a fusion exploration direction: leverage T3Former's temporal/topological modeling to enhance RouteNet's network prediction task.

**Key Questions:**
- Can T3Former's DoS/Betti topological features be migrated to RouteNet's traffic prediction task?
- Can T3Former's sliding window temporal modeling enhance RouteNet's perception of dynamic network traffic?
- Can the two modules' losses be jointly optimized?

---

## 5. Core Open Questions Checklist

### 5.1 New Fusion Model (Most Urgent)

| # | Category | Question |
|---|----------|---------|
| Q1 | Tokenizer | How are queue and link combined in HOP_TOKEN? Concatenation/addition/gating? Dim 64 or 32? |
| Q2 | Tokenizer | Variable path lengths: padding and mask strategy? |
| Q3 | Tokenizer | Does FLOW_TOKEN attend other flow tokens? Cross-graph handling? |
| Q4 | Tokenizer | Precise meaning of "flow as attention sliding window"? Transformer-XL analogy? |
| Q5 | Data | Flow definition: per-connection / per-packet? |
| Q6 | Data | Temporal granularity? Multi-timestep injection? |
| Q7 | Data | Dataset size? Train/val/test split? |
| Q8 | Topology | RoPE position definition: hop index or graph distance? |
| Q9 | Topology | Local GNN neighborhood definition: topological or sequential? GNN type? |
| Q10 | Topology | Value of x in "every x nodes as one learning group"? |
| Q11 | Architecture | Source of FFN expansion ratio 2.66? |
| Q12 | Architecture | Transformer layer count, head count, normalization type? |
| Q13 | Training | Optimizer, learning rate, dropout, batch size? |
| Q14 | Training | Use RouteNet-Fermi as baseline for comparison? |

### 5.2 Existing Module Optimization

| # | Category | Question |
|---|----------|---------|
| Q15 | RouteNet | Difference between all_mixed/ and real_traffic/? |
| Q16 | RouteNet | Few-shot definition? Generalization gap vs full training? |
| Q17 | T3Former | Is sliding window parameter tunable? |
| Q18 | T3Former | DoS + Betti feature calculation details need documentation |
| Q19 | Transfer | Can T3Former topological features transfer to RouteNet task? |

---

## 6. Project Directory Structure

```
graph DeepLearning/
|
+-- RouteNet-Fermi-main/                    [TensorFlow, Implemented]
|   +-- delay_model.py                      Core model (RouteNet_Fermi class)
|   +-- data/                               Datasets
|   |   +-- fat128/                         Fat-Tree 128 synthetic
|   |   +-- real_traces/                    Real networks (GEANT, Abilene...)
|   +-- traffic_models/                     By traffic model (delay/jitter/loss)
|   +-- scheduling/                         By scheduling policy
|   +-- scalability/                        Scale experiments
|   +-- all_mixed/                          Few-shot / full training
|   +-- real_traffic/                       Real traffic validation
|   +-- testbed/                            Testbed experiments
|   +-- fat_tree/                           Fat-Tree topology
|   +-- exercise/                           4-level learning exercises
|   +-- datanetAPI.py                       Network topology API
|
+-- T3former code/                          [PyTorch+PyG, Implemented]
|   +-- model.py                            Core models (T3Former, GNN, ...)
|   +-- train_T3Former.py                   Training script
|   +-- train_T3Former_neuro.py             Neuroscience dataset training
|   +-- data_loader.py                      Data loading
|   +-- modules.py                          Supporting modules
|   +-- dos_betti_features_*.py             Topological feature extraction
|   +-- TGN/                                Temporal Graph Networks
|   +-- TGAT/                               Temporal Graph Attention
|   +-- GraphMixer/                         GraphMixer implementation
|   +-- GCN_LSTM/                           GCN + LSTM baseline
|   +-- EvolveGCN/                          Evolving GCN
|   +-- NeuroGraph.py                       Neuroscience graph data
|
+-- text for analysis/                      [Documentation & Analysis]
|   +-- execution.md                        Fusion model implementation details
|   +-- The background analysis.md          Research background
|   +-- GraphiT paper reading report.md    Related paper notes
|   +-- Topics to learn.md                Learning roadmap
|   +-- Project Architecture Analysis Report.md  (this report)
|
+-- report/                                 [Existing analysis reports, 11 files]
|
+-- references/                             Reference materials
+-- mix architectutre/                      Mixed architecture experiments (typo in name)
+-- project text/                           Project texts
+-- .venv/                                 Virtual environment
```

---

## 7. Tech Stack Summary

| Module | Technology |
|--------|------------|
| RouteNet-Fermi | TensorFlow / Keras |
| T3Former | PyTorch + PyTorch Geometric |
| Graph processing | datanetAPI (RouteNet), PyG Data (T3Former) |
| Topological features | pyflagser (Betti), NumPy/SciPy (DoS) |
| Documentation | Markdown (Chinese) |

---

Report Status: Version 2 - Fully integrated RouteNet-Fermi and T3Former code analysis with systematic review of open questions in the fusion model. Needs confirmation on Q1-Q19 with project lead.
