# Level 2: 图神经网络消息传递练习

## 练习目标

理解图神经网络中的**消息传递**机制，这是 RouteNet-Fermi 的核心思想。

## 练习内容

本练习分为 3 个小节：

1. `exercise_2_1_simple_graph.py` - 理解简单的图结构表示
2. `exercise_2_2_message_passing.py` - 理解消息传递机制
3. `exercise_2_3_gru_message_passing.py` - 理解 GRU 消息传递

## RouteNet-Fermi 中的图结构

RouteNet-Fermi 将网络建模为三种节点的图：

| 节点类型 | 含义 | 示例 |
|----------|------|------|
| **Path (路径)** | 源到目的地的完整路由路径 | P0, P1, P2 |
| **Link (链路)** | 网络中的物理链路 | L0, L1, L2 |
| **Queue (队列)** | 链路上的优先级队列 | Q0, Q1, Q2 |

### 图的邻接关系

```
              ┌─────────────┐
              │     L0      │  (Link 0)
              └──────┬──────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       ┌──────┐  ┌──────┐  ┌──────┐
       │ Q0   │  │ Q1   │  │ Q2   │  (Queues)
       └──┬───┘  └──┬───┘  └──┬───┘
          │         │         │
          ▼         ▼         ▼
       ┌──────┐  ┌──────┐  ┌──────┐
       │ P0   │  │ P1   │  │ P2   │  (Paths)
       └──────┘  └──────┘  └──────┘
```

消息传递过程：
1. **Link/Queue → Path**: 聚合链路和队列信息更新路径状态
2. **Path → Queue**: 聚合路径信息更新队列状态
3. **Queue → Link**: 聚合队列信息更新链路状态

## 运行方式

```bash
cd exercise/level_2_message_passing
python exercise_2_1_simple_graph.py
python exercise_2_2_message_passing.py
python exercise_2_3_gru_message_passing.py
```

## 练习目标

1. **Exercise 2.1**: 理解图的邻接表表示和节点索引
2. **Exercise 2.2**: 理解基于聚合的消息传递
3. **Exercise 2.3**: 理解 GRU 作为消息更新函数

## 数据说明

使用简单的合成图数据：
- 3 个路径 (Paths)
- 2 个链路 (Links)
- 3 个队列 (Queues)
