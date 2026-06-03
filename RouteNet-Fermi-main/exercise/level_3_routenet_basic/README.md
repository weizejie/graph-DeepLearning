# Level 3: RouteNet-Fermi 基础结构

## 练习目标

理解 RouteNet-Fermi 的完整架构：
- 嵌入层：将原始网络特征转换为向量表示
- GRU 消息传递：迭代更新节点状态
- Readout 层：从路径状态预测性能指标

## 练习内容

本练习分为 2 个小节：

1. `exercise_3_1_embedding.py` - 理解特征嵌入层
2. `exercise_3_2_full_model.py` - 理解完整的 RouteNet-Fermi 模型

## RouteNet-Fermi 架构概览

```
输入特征:
├── Traffic (流量特征): traffic, packets, eq_lambda, ...
├── Path (路径特征): model (路由模型类型)
├── Link (链路特征): capacity, policy (调度策略)
└── Queue (队列特征): queue_size, priority, weight

        ↓
    嵌入层 (Embedding)
        ↓
    GRU 消息传递 (迭代 8 次)
    │
    ├─→ Link/Queue → Path
    ├─→ Path → Queue  
    └─→ Queue → Link
        ↓
    Readout 层
        ↓
    预测输出 (延迟/抖动/丢包)
```

## 运行方式

```bash
cd exercise/level_3_routenet_basic
python exercise_3_1_embedding.py
python exercise_3_2_full_model.py
```

## 练习目标

1. **Exercise 3.1**: 理解如何将原始网络特征转换为向量表示
2. **Exercise 3.2**: 理解完整的 RouteNet-Fermi 前向传播过程

## 数据说明

本级别使用简化的合成数据，特征结构与真实数据一致：
- Traffic 特征: 11维
- Path 特征: 7维 (one-hot 编码的 model)
- Link 特征: 5维 (one-hot 编码的 policy)
- Queue 特征: 5维 (queue_size + priority + weight)
