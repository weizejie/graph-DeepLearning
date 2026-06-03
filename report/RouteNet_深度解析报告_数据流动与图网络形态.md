# RouteNet 深度解析报告：数据流动与图网络形态

> 本报告从代码层面深入解析RouteNet的数据流动过程、超图的构建机制、消息传递的工作原理、训练结果的含义、参数更新机制，以及最终形成的图网络形态。

---

## 目录

1. [数据输入层：从网络拓扑到超图](#1-数据输入层从网络拓扑到超图)
2. [超图构建过程](#2-超图构建过程)
3. [嵌入层：特征向量化](#3-嵌入层特征向量化)
4. [GRU消息传递机制](#4-gru消息传递机制)
5. [预测层：延迟计算](#5-预测层延迟计算)
6. [完整数据流动图](#6-完整数据流动图)
7. [图网络形态总结](#7-图网络形态总结)
8. [关键设计洞察](#8-关键设计洞察)
9. [训练结果与参数更新机制](#9-训练结果与参数更新机制)

---

## 1. 数据输入层：从网络拓扑到超图

### 1.1 原始数据来源

```python
# data_generator.py
def generator(data_dir, shuffle, seed):
    tool = DatanetAPI(data_dir, shuffle=shuffle, seed=seed)
    it = iter(tool)
    for sample in it:
        G = nx.DiGraph(sample.get_topology_object())      # 网络拓扑
        T = sample.get_traffic_matrix()                    # 流量矩阵
        R = sample.get_routing_matrix()                    # 路由矩阵
        P = sample.get_performance_matrix()               # 性能矩阵
        HG = network_to_hypergraph(G=G, R=R, T=T, P=P)    # 构建超图
        yield hypergraph_to_input_data(HG)
```

### 1.2 三大核心输入

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RouteNet 三大输入                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  1. 拓扑矩阵 (Topology): G = nx.DiGraph()                                         │
│  ─────────────────────────────────────────                                        │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   节点属性:                                                                  │   │
│  │   • schedulingPolicy: 调度策略 (WFQ/SP/DRR/FIFO)                             │   │
│  │   • levelsQoS: QoS等级数量                                                   │   │
│  │   • bufferSizes: 队列大小                                                    │   │
│  │                                                                             │   │
│  │   边属性:                                                                    │   │
│  │   • bandwidth: 链路带宽                                                      │   │
│  │   • port: 端口号                                                             │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  2. 流量矩阵 (Traffic Matrix): T[src, dst]                                        │
│  ─────────────────────────────────────────                                        │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   流量特征 (每个Flow):                                                       │   │
│  │   • traffic/AvgBw: 平均带宽 (kbps)                                          │   │
│  │   • packets/PktsGen: 生成包数                                               │   │
│  │   • TimeDist: 时间分布类型 (EXP/UNIF/NORMAL/ONOFF/PPBP/TRACE)               │   │
│  │   • ToS: 服务类型                                                            │   │
│  │                                                                             │   │
│  │   时间分布参数:                                                              │   │
│  │   • eq_lambda: 等效到达率                                                    │   │
│  │   • avg_pkts_lambda: 平均包到达率                                           │   │
│  │   • exp_max_factor: 指数最大因子                                            │   │
│  │   • pkts_lambda_on: 开状态包到达率                                          │   │
│  │   • avg_t_off: 平均关闭时间                                                  │   │
│  │   • avg_t_on: 平均开启时间                                                   │   │
│  │   • ar_a: 自回归参数                                                        │   │
│  │   • sigma: 标准差                                                          │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  3. 路由矩阵 (Routing Matrix): R[src, dst]                                         │
│  ─────────────────────────────────────────                                         │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   路径表示:                                                                  │   │
│  │   R[src, dst] = [node_0, node_1, node_2, ..., node_k]                     │   │
│  │                                                                             │   │
│  │   示例:                                                                      │   │
│  │   R[0, 5] = [0, 2, 4, 5]  # 节点0到节点5的路径: 0→2→4→5                 │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 超图构建过程

### 2.1 网络到超图的转换

超图构建是RouteNet的核心创新之一，它将原始网络转换为包含三种节点类型的异构图：

```python
# data_generator.py - network_to_hypergraph 函数

def network_to_hypergraph(G, R, T, P):
    D_G = nx.DiGraph()
    
    # 遍历所有源-目的节点对
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst and G.has_edge(src, dst):
                # 创建链路节点
                D_G.add_node('l_{}_{}'.format(src, dst),
                             capacity=G.edges[src, dst]['bandwidth'],
                             policy=np.where(G.edges[src, dst]['schedulingPolicy'] == POLICIES)[0][0])
                
                # 遍历所有流
                for f_id in range(len(T[src, dst]['Flows'])):
                    # 创建Path节点 (p = path)
                    D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),
                                 source=src, destination=dst,
                                 traffic=T[src, dst]['Flows'][f_id]['AvgBw'],
                                 packets=T[src, dst]['Flows'][f_id]['PktsGen'],
                                 model=flow['TimeDist'].value,
                                 delay=P[src, dst]['Flows'][f_id]['AvgDelay'])
                    
                    # 沿路径创建队列节点和边
                    for h_1, h_2 in [R[src, dst][i:i+2] for i in range(len(R[src, dst])-1)]:
                        # Path ↔ Link 双向边
                        D_G.add_edge('l_{}_{}'.format(h_1, h_2), 
                                     'p_{}_{}_{}'.format(src, dst, f_id))
                        D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id),
                                     'l_{}_{}'.format(h_1, h_2))
                        
                        # 创建队列节点
                        for q in range(G.nodes[h_1]['levelsQoS']):
                            D_G.add_node('q_{}_{}_{}'.format(h_1, h_2, q),
                                         queue_size=q_s[q],
                                         priority=q,
                                         weight=q_w[q] if q_w[0] != '-' else 0)
                            
                            # Link ↔ Queue 边
                            D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q),
                                         'l_{}_{}'.format(h_1, h_2))
                            
                            # Path ↔ Queue 边 (根据ToS映射)
                            if str(int(T[src, dst]['Flows'][0]['ToS'])) in q_map[q]:
                                D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id),
                                             'q_{}_{}_{}'.format(h_1, h_2, q))
                                D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q),
                                             'p_{}_{}_{}'.format(src, dst, f_id))
```

### 2.2 三层异构节点结构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RouteNet 超图结构                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                              RouteNet 异构超图                                       │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                               │ │
│  │                                                                               │ │
│  │                        ┌─────────────┐                                        │ │
│  │                        │  Path节点   │                                        │ │
│  │                        │  (p_src_dst)│                                        │ │
│  │                        └──────┬──────┘                                        │ │
│  │                               │                                               │ │
│  │           ┌───────────────────┼───────────────────┐                           │ │
│  │           │                   │                   │                           │ │
│  │           ▼                   ▼                   ▼                           │ │
│  │     ┌──────────┐        ┌──────────┐        ┌──────────┐                    │ │
│  │     │ Link节点  │◄─────►│Queue节点 │◄─────►│ Link节点 │                    │ │
│  │     │ (l_h1_h2)│        │(q_h1_h2_q)│        │ (l_h2_h3)│                    │ │
│  │     └────┬─────┘        └──────────┘        └────┬─────┘                    │ │
│  │          │                                      │                            │ │
│  │          │         RouteNet三层结构              │                            │ │
│  │          │                                      │                            │ │
│  │          └──────────────────────────────────────┘                            │ │
│  │                          Layer 1: Link                                       │ │
│  │                                                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│  节点类型详解:                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                               │ │
│  │  Path节点 (p): 端到端的流                                                    │ │
│  │  ─────────────────────────────────────────                                   │ │
│  │  • 标识: p_src_dst_flow_id                                                  │ │
│  │  • 属性: traffic, packets, model, delay                                      │ │
│  │  • 数量: N×(N-1)×F (N=节点数, F=平均流数)                                   │ │
│  │                                                                               │ │
│  │  Link节点 (l): 网络链路                                                      │ │
│  │  ─────────────────────────────────                                          │ │
│  │  • 标识: l_node1_node2                                                      │ │
│  │  • 属性: capacity, policy                                                   │ │
│  │  • 数量: E (边数)                                                           │ │
│  │                                                                               │ │
│  │  Queue节点 (q): 路由器队列                                                   │ │
│  │  ─────────────────────────────────                                          │ │
│  │  • 标识: q_node1_node2_queue_id                                             │ │
│  │  • 属性: queue_size, priority, weight                                        │ │
│  │  • 数量: E × levelsQoS                                                      │ │
│  │                                                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 关系矩阵的构建

```python
# 从超图构建关系矩阵

# 超图到输入数据的转换
def hypergraph_to_input_data(HG):
    # 收集五种关键关系
    link_to_path = []      # Link → Path: 哪些路径经过该链路
    queue_to_path = []     # Queue → Path: 哪些路径经过该队列
    path_to_queue = []     # Path → Queue: 路径经过哪些队列
    queue_to_link = []     # Queue → Link: 队列属于哪个链路
    path_to_link = []      # Path → Link: 路径经过哪些链路
    
    # 遍历超图节点构建关系
    for node in HG.nodes:
        in_nodes = [s for s, d in HG.in_edges(node)]
        if node.startswith('q_'):  # Queue节点
            # 找到经过该队列的所有路径
            path = []
            for n in in_nodes:
                if n.startswith('p_'):
                    path.append([int(n.replace('p_', '')), ...])
            path_to_queue.append(path)
        elif node.startswith('p_'):  # Path节点
            # 收集该路径经过的链路和队列
            links = [int(n.replace('l_', '')) for n in in_nodes if n.startswith('l_')]
            queues = [int(n.replace('q_', '')) for n in in_nodes if n.startswith('q_')]
            link_to_path.append(links)
            queue_to_path.append(queues)
        elif node.startswith('l_'):  # Link节点
            # 收集该链路上的队列和路径
            queues = [int(n.replace('q_', '')) for n in in_nodes if n.startswith('q_')]
            paths = [[int(n.replace('p_', '')), ...] for n in in_nodes if n.startswith('p_')]
            queue_to_link.append(queues)
            path_to_link.append(paths)
```

### 2.4 关系矩阵的数据结构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              关系矩阵数据结构                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  使用 tf.ragged.constant 表示不规则关系:                                            │
│                                                                                     │
│  link_to_path:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   link_to_path[link_id] = [path_id_1, path_id_2, ...]                      │   │
│  │   表示: link_id 经过的路径列表                                               │   │
│  │                                                                             │   │
│  │   示例:                                                                      │   │
│  │   link_to_path = [[0, 1],      # Link 0 经过 Path 0, 1                       │   │
│  │                   [1, 2]]      # Link 1 经过 Path 1, 2                       │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  queue_to_path:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   queue_to_path[queue_id] = [path_id_1, path_id_2, ...]                     │   │
│  │   表示: queue_id 经过的路径列表                                               │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  path_to_queue:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   path_to_queue[path_id] = [[queue_id, position_in_path], ...]            │   │
│  │   表示: path_id 经过的队列列表及其在路径中的位置                             │   │
│  │                                                                             │   │
│  │   示例:                                                                      │   │
│  │   path_to_queue = [[[0, 0]],        # Path 0 经过 Queue 0，位置0           │   │
│  │                    [[1, 0], [2, 1]], # Path 1 经过 Queue 1,2，位置0,1     │   │
│  │                    [[2, 0]]]        # Path 2 经过 Queue 2，位置0           │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 嵌入层：特征向量化

### 3.1 嵌入层架构

```python
# delay_model.py

class RouteNet_Fermi(tf.keras.Model):
    def __init__(self):
        # 状态维度
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32
        
        # Path嵌入: 10个数值特征 + 7个流量模型one-hot = 17维 → 32维
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=10 + 7),  # 17维输入
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        
        # Link嵌入: capacity(1) + policy one-hot(4) = 5维 → 32维
        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=4 + 1),   # 5维输入
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        
        # Queue嵌入: queue_size(1) + priority one-hot(3) + weight(1) = 5维 → 32维
        self.queue_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=3 + 1 + 1),  # 5维输入
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
```

### 3.2 特征标准化与嵌入

```python
# 初始化路径状态
path_state = self.path_embedding(tf.concat(
    # 10个数值特征 (Z-Score标准化)
    [(traffic - mean) / std,      # 流量
     (packets - mean) / std,      # 包数
     (eq_lambda - mean) / std,    # 等效到达率
     (avg_pkts_lambda - mean) / std,  # 平均包到达率
     (exp_max_factor - mean) / std,  # 指数最大因子
     (pkts_lambda_on - mean) / std,  # 开状态包到达率
     (avg_t_off - mean) / std,    # 平均关闭时间
     (avg_t_on - mean) / std,     # 平均开启时间
     (ar_a - mean) / std,         # 自回归参数
     (sigma - mean) / std,        # 标准差
     
     # 1个分类特征 (one-hot编码)
     tf.one_hot(model, 7)          # 流量模型类型
    ], axis=1))
```

### 3.3 嵌入层维度详解

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              嵌入层维度详解                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  Path嵌入:                                                                         │
│  ────────                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                               │ │
│  │   输入维度: 17                                                                │ │
│  │   ├── 数值特征: 10 (traffic, packets, eq_lambda, ...)                         │ │
│  │   └── 分类特征: 7 (流量模型 one-hot)                                          │ │
│  │                                                                               │ │
│  │   输出维度: 32 (path_state_dim)                                               │ │
│  │                                                                               │ │
│  │   形状: [batch, num_paths, 32]                                               │ │
│  │                                                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│  Link嵌入:                                                                         │
│  ────────                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                               │ │
│  │   输入维度: 5                                                                 │ │
│  │   ├── capacity: 1 (链路容量)                                                  │ │
│  │   └── policy: 4 (调度策略 one-hot: WFQ/SP/DRR/FIFO)                          │ │
│  │                                                                               │ │
│  │   输出维度: 32 (link_state_dim)                                               │ │
│  │                                                                               │ │
│  │   形状: [batch, num_links, 32]                                               │ │
│  │                                                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│  Queue嵌入:                                                                        │
│  ──────────                                                                        │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                               │ │
│  │   输入维度: 5                                                                 │ │
│  │   ├── queue_size: 1 (队列大小)                                                │ │
│  │   ├── priority: 3 (优先级 one-hot: 0/1/2)                                    │ │
│  │   └── weight: 1 (权重)                                                       │ │
│  │                                                                               │ │
│  │   输出维度: 32 (queue_state_dim)                                              │ │
│  │                                                                               │ │
│  │   形状: [batch, num_queues, 32]                                              │ │
│  │                                                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. GRU消息传递机制

### 4.1 消息传递的核心循环

```python
# delay_model.py - 核心消息传递循环

# GRU单元定义
self.path_update = tf.keras.layers.GRUCell(32)
self.link_update = tf.keras.layers.GRUCell(32)
self.queue_update = tf.keras.layers.GRUCell(32)

# 迭代8次的消息传递
for it in range(self.iterations):  # iterations = 8
    
    # ========== 步骤1: Link + Queue → Path ==========
    # 聚合链路和队列信息，更新路径状态
    queue_gather = tf.gather(queue_state, queue_to_path)
    link_gather = tf.gather(link_state, link_to_path)
    
    # 使用RNN逐步处理路径上的每一跳
    path_state_sequence, path_state = path_update_rnn(
        tf.concat([queue_gather, link_gather], axis=2),
        initial_state=path_state
    )
    
    # ========== 步骤2: Path → Queue ==========
    # 路径信息传递给队列
    path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
    path_sum = tf.math.reduce_sum(path_gather, axis=1)
    queue_state, _ = self.queue_update(path_sum, [queue_state])
    
    # ========== 步骤3: Queue → Link ==========
    # 队列信息传递给链路
    queue_gather = tf.gather(queue_state, queue_to_link)
    link_state = link_gru_rnn(queue_gather, initial_state=link_state)
```

### 4.2 消息传递的可视化流程

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         RouteNet 8次迭代消息传递                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  迭代 t = 0, 1, 2, ..., 7:                                                        │
│  ─────────────────────────────────                                                │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │   │                                                                     │ │   │
│  │   │   步骤1: Link + Queue ──► Path (RNN处理跳序列)                      │ │   │
│  │   │   ───────────────────────────────────────────                        │ │   │
│  │   │                                                                     │ │   │
│  │   │   queue_gather = gather(queue_state, queue_to_path)                 │ │   │
│  │   │   link_gather = gather(link_state, link_to_path)                   │ │   │
│  │   │                                                                     │ │   │
│  │   │   # RNN展开 (假设路径有K跳):                                         │ │   │
│  │   │   h₀ = Initial(Path嵌入)                                            │ │   │
│  │   │   h₁ = GRU(h₀, [q₀, l₀])                                          │ │   │
│  │   │   h₂ = GRU(h₁, [q₁, l₁])                                          │ │   │
│  │   │   ...                                                               │ │   │
│  │   │   h_K = GRU(h_{K-1}, [q_{K-1}, l_{K-1}])                          │ │   │
│  │   │                                                                     │ │   │
│  │   │   path_state_sequence = [h₀, h₁, ..., h_K]                        │ │   │
│  │   │   path_state = h_K  # 最终路径状态                                  │ │   │
│  │   │                                                                     │ │   │
│  │   └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                               │                                             │   │
│  │                               ▼                                             │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │   │                                                                     │ │   │
│  │   │   步骤2: Path ──► Queue (路径影响队列)                              │ │   │
│  │   │   ───────────────────────────────────────                            │ │   │
│  │   │                                                                     │ │   │
│  │   │   path_gather = gather(path_state_sequence, path_to_queue)          │ │   │
│  │   │   path_sum = sum(path_gather, axis=1)                              │ │   │
│  │   │   queue_state = GRU(path_sum, queue_state)                          │ │   │
│  │   │                                                                     │ │   │
│  │   │   # 队列聚合所有经过它的路径信息                                     │ │   │
│  │   │                                                                     │ │   │
│  │   └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                               │                                             │   │
│  │                               ▼                                             │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │   │                                                                     │ │   │
│  │   │   步骤3: Queue ──► Link (队列影响链路)                              │ │   │
│  │   │   ───────────────────────────────────────                            │ │   │
│  │   │                                                                     │ │   │
│  │   │   queue_gather = gather(queue_state, queue_to_link)                 │ │   │
│  │   │   link_state = RNN(queue_gather, initial_state=link_state)          │ │   │
│  │   │                                                                     │ │   │
│  │   │   # 链路聚合所有相连队列的信息                                       │ │   │
│  │   │                                                                     │ │   │
│  │   └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 跳序列处理的细节

```python
# 跳序列处理的核心代码

# queue_to_path: 路径经过的队列索引
# 例如: queue_to_path = [[0, 1, 2], [3, 4], [5]] 表示3条路径分别经过队列[0,1,2], [3,4], [5]

# gather操作: 按路径收集队列状态
queue_gather = tf.gather(queue_state, queue_to_path)

# 结果形状: [num_paths, max_path_length, queue_state_dim]
# 不规则张量 (RaggedTensor)

# link_gather: 按路径收集链路状态
link_gather = tf.gather(link_state, link_to_path)

# 合并Queue和Link信息
combined = tf.concat([queue_gather, link_gather], axis=2)
# 形状: [num_paths, max_path_length, queue_state_dim + link_state_dim]

# RNN处理序列
path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                      return_sequences=True,  # 返回所有时间步的输出
                                      return_state=True)

path_state_sequence, path_state = path_update_rnn(combined, initial_state=path_state)
# path_state_sequence: [num_paths, path_length, path_state_dim] - 每跳的隐藏状态
# path_state: [num_paths, path_state_dim] - 最终隐藏状态
```

### 4.4 消息传递的物理意义

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         消息传递的物理意义                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  Link → Queue → Path 的信息流动模拟了网络的排队论动态                              │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   物理过程:                                                                  │   │
│  │   ─────────                                                                  │   │
│  │                                                                             │   │
│  │   1. 链路负载 (load) = Σ(经过路径的流量) / 链路容量                        │   │
│  │      这是RouteNet计算的第一件事:                                             │   │
│  │      load = sum(path_gather_traffic) / capacity                             │   │
│  │                                                                             │   │
│  │   2. 链路状态 → 队列服务率                                                   │   │
│  │      • 链路带宽决定队列的出队速率                                           │   │
│  │      • 调度策略(WFQ/FIFO等)影响多队列的服务公平性                          │   │
│  │                                                                             │   │
│  │   3. 队列状态 → 路径排队延迟                                                 │   │
│  │      • 队列长度 = f(到达率, 服务率)                                         │   │
│  │      • 排队延迟 ≈ 队列占用 / 服务率                                          │   │
│  │                                                                             │   │
│  │   4. 路径状态 → 端到端延迟                                                   │   │
│  │      • 端到端延迟 = Σ(每跳排队延迟) + Σ(传输延迟)                           │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  GRU的门控机制作用:                                                                │
│  ───────────────────                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   GRU的更新门和重置门决定:                                                   │   │
│  │   • 保留多少之前的状态信息                                                  │   │
│  │   • 接收多少当前跳的信息                                                     │   │
│  │                                                                             │   │
│  │   这类似于排队论中的"记忆效应":                                             │   │
│  │   • 之前的队列状态影响当前的排队行为                                        │   │
│  │   • 但不会完全记忆所有历史                                                  │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  8次迭代的作用:                                                                   │
│  ───────────────                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   迭代过程逐步逼近网络平衡态:                                                │   │
│  │                                                                             │   │
│  │   迭代1: 初始估计                                                           │   │
│  │   迭代2: 考虑路径间竞争                                                     │   │
│  │   迭代3: 考虑链路拥塞反馈                                                   │   │
│  │   ...                                                                       │   │
│  │   迭代8: 充分收敛                                                           │   │
│  │                                                                             │   │
│  │   8次迭代后:                                                                │   │
│  │   • 路径状态包含了整条路径的累积信息                                        │   │
│  │   • 队列状态反映了所有经过流量的竞争                                        │   │
│  │   • 链路状态编码了负载和调度策略的影响                                      │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 预测层：延迟计算

### 5.1 预测的核心公式

```python
# delay_model.py - 延迟预测

# 收集路径经过的链路容量
capacity_gather = tf.gather(capacity, link_to_path)

# occupancy预测 (从路径状态序列)
input_tensor = path_state_sequence[:, 1:].to_tensor()  # 去掉初始状态h₀
occupancy_gather = self.readout_path(input_tensor)      # [num_paths, path_len, 1]

# 处理不规则张量
length = tf.ensure_shape(length, [None])
occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

# 排队延迟 = Σ(occupancy / capacity)
queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)

# 传输延迟 = Σ(pkt_size / capacity) = Σ(packets / capacity) × avg_pkt_size
pkt_size = traffic / packets
trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

# 总延迟
return queue_delay + trans_delay
```

### 5.2 延迟计算详解

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         延迟预测详解                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  总延迟 = 排队延迟 + 传输延迟                                                      │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   1. 排队延迟 (Queueing Delay)                                             │   │
│  │   ─────────────────────────────                                             │   │
│  │                                                                             │   │
│  │   核心思想: 排队延迟 ≈ 队列占用 / 服务率                                      │   │
│  │                                                                             │   │
│  │   步骤:                                                                     │   │
│  │   a) occupancy预测: occupancy = Readout(path_state_i)                       │   │
│  │      • path_state_i: 第i跳处理后的隐藏状态                                  │   │
│  │      • Readout: MLP网络                                                     │   │
│  │                                                                             │   │
│  │   b) 每跳延迟: delay_i = occupancy_i / capacity_i                          │   │
│  │                                                                             │   │
│  │   c) 总排队延迟: queue_delay = Σ(delay_i)                                  │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   2. 传输延迟 (Transmission Delay)                                         │   │
│  │   ─────────────────────────────────                                        │   │
│  │                                                                             │   │
│  │   核心思想: 传输延迟 = 包大小 / 链路带宽                                     │   │
│  │                                                                             │   │
│  │   公式:                                                                     │   │
│  │   pkt_size = traffic / packets = avg_pkt_size                               │   │
│  │   trans_delay = pkt_size × Σ(1 / capacity_i)                               │   │
│  │                                                                             │   │
│  │   简化理解:                                                                 │   │
│  │   trans_delay = Σ(pkt_size / capacity_i)                                   │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Readout网络结构

```python
# Readout网络定义
self.readout_path = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 32)),  # 输入: 路径状态序列
    tf.keras.layers.Dense(16, activation='relu'),   # 32 → 16
    tf.keras.layers.Dense(16, activation='relu'),   # 16 → 16
    tf.keras.layers.Dense(1)                       # 16 → 1 (occupancy预测)
])
```

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Readout网络详解                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  Readout的输入: path_state_sequence                                                │
│  ─────────────────────────────────────────                                        │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   path_state_sequence[:, 1:]:                                               │   │
│  │   • 形状: [num_paths, path_length, 32]                                      │   │
│  │   • 去掉h₀，保留[h₁, h₂, ..., h_K]                                        │   │
│  │   • 每跳的隐藏状态代表该跳之后的状态                                         │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  Readout的输出: occupancy预测                                                      │
│  ───────────────────────────                                                       │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   occupancy_gather:                                                         │   │
│  │   • 形状: [num_paths, path_length, 1]                                       │   │
│  │   • 每个时间步预测一个occupancy值                                           │   │
│  │                                                                             │   │
│  │   物理意义:                                                                  │   │
│  │   • occupancy_i = Readout(h_i)                                             │   │
│  │   • 表示: "在第i跳之后，队列预计占用的资源"                                 │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  为什么去掉h₀?                                                                     │
│  ───────────────                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   h₀是初始路径嵌入，还没经过任何队列                                        │   │
│  │   h₁是经过第1跳后的状态，包含了第1跳的队列信息                              │   │
│  │   所以应该从h₁开始预测                                                       │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 完整数据流动图

### 6.1 端到端数据流动

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RouteNet 完整数据流动                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  输入层                                                                             │
│  ──────                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   Topology (G)      Traffic (T)       Routing (R)       Performance (P)     │   │
│  │       │                 │                 │                  │              │   │
│  │       │                 │                 │                  │              │   │
│  │       ▼                 ▼                 ▼                  ▼              │   │
│  │   节点属性:         流量特征:          路径信息:           延迟标签:        │   │
│  │   • 调度策略        • traffic          • R[src,dst]       • AvgDelay       │   │
│  │   • 队列大小        • packets          • 跳序列           • Jitter         │   │
│  │   • QoS等级         • model            • 长度             • Losses         │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                              │
│                                     ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   超图构建 (network_to_hypergraph)                                           │   │
│  │   ─────────────────────────────────────────                                  │   │
│  │                                                                             │   │
│  │   原始网络 ──► 三层异构图                                                    │   │
│  │                                                                             │   │
│  │       ├── Path节点: p_src_dst_flow_id                                       │   │
│  │       ├── Link节点: l_node1_node2                                           │   │
│  │       └── Queue节点: q_node1_node2_queue_id                                │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                              │
│                                     ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   关系矩阵构建 (hypergraph_to_input_data)                                    │   │
│  │   ──────────────────────────────────────────                                 │   │
│  │                                                                             │   │
│  │   ├── link_to_path: Link → Path                                            │   │
│  │   ├── queue_to_path: Queue → Path                                           │   │
│  │   ├── path_to_queue: Path → Queue (带位置)                                  │   │
│  │   ├── queue_to_link: Queue → Link                                           │   │
│  │   └── path_to_link: Path → Link (带位置)                                   │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                              │
│                                     ▼                                              │
│  嵌入层                                                                             │
│  ──────                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │   │
│  │   │ Path嵌入    │     │ Link嵌入    │     │ Queue嵌入   │                   │   │
│  │   │             │     │             │     │             │                   │   │
│  │   │ 17维→32维   │     │ 5维→32维    │     │ 5维→32维    │                   │   │
│  │   │             │     │             │     │             │                   │   │
│  │   │ Z-Score +   │     │ capacity +  │     │ queue_size +│                   │   │
│  │   │ one-hot     │     │ policy      │     │ priority +  │                   │   │
│  │   │             │     │             │     │ weight      │                   │   │
│  │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │   │
│  │          │                   │                   │                            │   │
│  │          ▼                   ▼                   ▼                            │   │
│  │   path_state (32D)   link_state (32D)    queue_state (32D)                 │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                              │
│                                     ▼                                              │
│  消息传递层 (×8迭代)                                                                 │
│  ─────────────────────                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   迭代 1:                                                                │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │   │                                                                     │ │   │
│  │   │   Link + Queue ──► Path (RNN)                                        │ │   │
│  │   │   path_state = GRU(path_state, [queue_gather, link_gather])        │ │   │
│  │   │                                                                     │ │   │
│  │   │   Path ──► Queue                                                    │ │   │
│  │   │   queue_state = GRU(queue_state, path_sum)                          │ │   │
│  │   │                                                                     │ │   │
│  │   │   Queue ──► Link                                                    │ │   │
│  │   │   link_state = GRU(link_state, queue_gather)                       │ │   │
│  │   │                                                                     │ │   │
│  │   └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                     │                                     │   │
│  │   迭代 2: ──────────────────────────┘                                      │   │
│  │   ...                                                                     │   │
│  │   迭代 8: ──────────────────────────┘ (充分收敛)                          │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                              │
│                                     ▼                                              │
│  输出层                                                                             │
│  ──────                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │   │                                                                     │ │   │
│  │   │   Readout (occupancy预测)                                          │ │   │
│  │   │   occupancy = MLP(path_state_sequence[:, 1:])                      │ │   │
│  │   │                                                                     │ │   │
│  │   └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                     │                                     │   │
│  │                                     ▼                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │   │                                                                     │ │   │
│  │   │   延迟计算                                                          │ │   │
│  │   │                                                                     │ │   │
│  │   │   queue_delay = Σ(occupancy / capacity)                            │ │   │
│  │   │   trans_delay = pkt_size × Σ(1 / capacity)                        │ │   │
│  │   │                                                                     │ │   │
│  │   │   delay = queue_delay + trans_delay                                 │ │   │
│  │   │                                                                     │ │   │
│  │   └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                              │
│                                     ▼                                              │
│                               延迟预测值                                            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 图网络形态总结

### 7.1 最终图网络的三层结构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RouteNet 最终图网络                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                              最终表示形态                                           │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   Layer 3: Path节点层                                                       │   │
│  │   ───────────────────                                                        │   │
│  │                                                                             │   │
│  │   节点数量: N × (N-1) × F  (N=节点数, F=每对平均流数)                        │   │
│  │   节点表示: 32维向量 (path_state)                                           │   │
│  │   节点特征: traffic, packets, model, eq_lambda, ...                         │   │
│  │                                                                             │   │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                │   │
│  │   │ p₁  │ │ p₂  │ │ p₃  │ │ p₄  │ │ ... │  ← 每条流一个节点               │   │
│  │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                                │   │
│  │      │       │       │       │       │                                     │   │
│  └──────┼───────┼───────┼───────┼───────┼─────────────────────────────────────┘   │
│         │       │       │       │       │                                       │
│  ┌──────┼───────┼───────┼───────┼───────┼─────────────────────────────────────┐   │
│  │      ▼       ▼       ▼       ▼       ▼                                       │   │
│  │                                                                               │   │
│  │   Layer 2: Queue节点层                                                        │   │
│  │   ────────────────────                                                        │   │
│  │                                                                               │   │
│  │   节点数量: E × Q  (E=边数, Q=平均QoS等级)                                   │   │
│  │   节点表示: 32维向量 (queue_state)                                           │   │
│  │   节点特征: queue_size, priority, weight                                     │   │
│  │                                                                               │   │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │   │
│  │   │ q₁  │ │ q₂  │ │ q₃  │ │ q₄  │ │ q₅  │ │ q₆  │ │ q₇  │ │ ... │        │   │
│  │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘        │   │
│  │      │       │       │       │       │       │       │       │              │   │
│  └──────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼──────────────┘   │
│         │       │       │       │       │       │       │       │                  │
│  ┌──────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────────┐   │
│  │      ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼                │   │
│  │                                                                               │   │
│  │   Layer 1: Link节点层                                                         │   │
│  │   ──────────────────                                                          │   │
│  │                                                                               │   │
│  │   节点数量: E  (网络边数)                                                     │   │
│  │   节点表示: 32维向量 (link_state)                                             │   │
│  │   节点特征: capacity, policy                                                  │   │
│  │                                                                               │   │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                  │   │
│  │   │ l₁  │ │ l₂  │ │ l₃  │ │ l₄  │ │ ... │  ← 每条链路一个节点               │   │
│  │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                                   │   │
│  │                                                                               │   │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 边类型与消息方向

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              边类型与消息方向                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  RouteNet的三种边 (无向边，双向消息传递):                                           │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   边类型1: Path ↔ Queue (排队影响)                                          │   │
│  │   ─────────────────────────────────                                        │   │
│  │                                                                             │   │
│  │   Path ──► Queue: 流量到达影响队列状态                                      │   │
│  │   Queue ──► Path: 排队延迟影响路径状态                                       │   │
│  │                                                                             │   │
│  │   特点: 沿路径方向，顺序传递                                                 │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   边类型2: Queue ↔ Link (服务与负载)                                        │   │
│  │   ─────────────────────────────────                                        │   │
│  │                                                                             │   │
│  │   Link ──► Queue: 链路服务能力影响队列出队                                  │   │
│  │   Queue ──► Link: 队列占用反映链路负载                                       │   │
│  │                                                                             │   │
│  │   特点: 本地连接，紧密耦合                                                   │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │   边类型3: Path ↔ Link (直接连接)                                          │   │
│  │   ─────────────────────────────────                                        │   │
│  │                                                                             │   │
│  │   Path ──► Link: 流经链路的流量贡献负载                                     │   │
│  │   Link ──► Path: 链路容量影响传输延迟                                        │   │
│  │                                                                             │   │
│  │   特点: 提供直接的消息通道                                                  │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 关键设计洞察

### 8.1 RouteNet如何缓解GNN固有问题

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         RouteNet vs 传统GNN                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  问题1: 过平滑 (Over-smoothing)                                                    │
│  ────────────────────────────────                                                  │
│                                                                                     │
│  传统GNN:                                                                          │
│  • 深层聚合导致所有节点表示趋于相同                                                │
│                                                                                     │
│  RouteNet:                                                                          │
│  • 每条路径独立处理，不做全图平均化                                                │
│  • RNN保持每跳的差异化信息                                                         │
│  • 路径长度固定了迭代次数                                                         │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  问题2: 过压缩 (Over-squashing)                                                    │
│  ─────────────────────────────                                                      │
│                                                                                     │
│  传统GNN:                                                                          │
│  • 大量邻居信息被压缩到固定维度                                                    │
│                                                                                     │
│  RouteNet:                                                                          │
│  • GRU门控机制选择保留重要信息                                                     │
│  • 8次小范围迭代比一次大范围聚合更有效                                             │
│  • 路径结构天然限制了邻居数量                                                      │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  问题3: 感受野有限                                                                │
│  ─────────────────                                                                  │
│                                                                                     │
│  传统GNN:                                                                          │
│  • K层只能看到K跳内的信息                                                         │
│                                                                                     │
│  RouteNet:                                                                          │
│  • 8次迭代扩展感受野                                                              │
│  • 每次迭代信息沿路径传播                                                         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 RouteNet的独特优势

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         RouteNet 独特设计优势                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  优势1: 拓扑零样本泛化                                                            │
│  ─────────────────────                                                              │
│                                                                                     │
│  • 基于归纳偏置设计，不依赖特定拓扑                                                 │
│  • 可以泛化到训练时未见过的网络拓扑                                                │
│  • 关键: 邻接关系通过关系矩阵定义，与拓扑无关                                      │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  优势2: 物理可解释性                                                              │
│  ─────────────────────                                                              │
│                                                                                     │
│  • 排队延迟 = occupancy / capacity  ← M/M/1排队论公式                              │
│  • 传输延迟 = pkt_size / capacity  ← 确定性公式                                    │
│  • Readout预测occupancy，物理意义清晰                                              │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  优势3: 异构图显式建模                                                            │
│  ─────────────────────                                                              │
│                                                                                     │
│  • Path/Link/Queue三种节点类型独立嵌入                                             │
│  • 关系矩阵显式定义连接                                                           │
│  • 比通用异构图方法更针对网络建模                                                 │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  优势4: 流量模型的充分利用                                                        │
│  ─────────────────────                                                              │
│                                                                                     │
│  • 7种流量模型one-hot编码为特征                                                   │
│  • 模型可学习不同流量模型的特性                                                   │
│  • 包含时序参数(eq_lambda, ON/OFF时间等)                                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 数据流动的数学表达

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         数据流动的数学表达                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  符号定义:                                                                          │
│  ────────                                                                          │
│                                                                                     │
│  P: Path节点数量, Q: Queue节点数量, L: Link节点数量                                │
│  d: 状态维度 = 32                                                                 │
│  K: 路径长度 (跳数)                                                               │
│  T: 迭代次数 = 8                                                                 │
│                                                                                     │
│  嵌入:                                                                             │
│  ────                                                                              │
│                                                                                     │
│  h_P⁰ = Embed_P(traffic, packets, model, ...)  ∈ ℝ^{P×d}                         │
│  h_Q⁰ = Embed_Q(queue_size, priority, weight)  ∈ ℝ^{Q×d}                         │
│  h_L⁰ = Embed_L(capacity, policy)  ∈ ℝ^{L×d}                                     │
│                                                                                     │
│  消息传递 (迭代 t=1,...,T):                                                       │
│  ───────────────────────────                                                       │
│                                                                                     │
│  Step 1: Link + Queue → Path                                                       │
│  ────────────────────────────                                                       │
│  q_gather = gather(h_Q^{t-1}, queue_to_path)  ∈ ℝ^{P×K×d}                        │
│  l_gather = gather(h_L^{t-1}, link_to_path)  ∈ ℝ^{P×K×d}                        │
│  combined = concat([q_gather, l_gather], axis=-1)  ∈ ℝ^{P×K×2d}                 │
│  h_P^t = RNN(combined, h_P^{t-1})  ∈ ℝ^{P×d}                                    │
│                                                                                     │
│  Step 2: Path → Queue                                                              │
│  ─────────────────────                                                              │
│  p_gather = gather(h_P^t, path_to_queue)  ∈ ℝ^{Q×?}                               │
│  p_sum = sum(p_gather, axis=1)  ∈ ℝ^{Q×d}                                        │
│  h_Q^t = GRU(p_sum, h_Q^{t-1})  ∈ ℝ^{Q×d}                                        │
│                                                                                     │
│  Step 3: Queue → Link                                                              │
│  ─────────────────────                                                              │
│  q_gather = gather(h_Q^t, queue_to_link)  ∈ ℝ^{L×?}                               │
│  h_L^t = RNN(q_gather, h_L^{t-1})  ∈ ℝ^{L×d}                                    │
│                                                                                     │
│  预测:                                                                              │
│  ────                                                                              │
│                                                                                     │
│  occupancy = Readout(h_P^T[:, 1:])  ∈ ℝ^{P×K×1}                                 │
│  queue_delay = Σ(occupancy / capacity)  ∈ ℝ^P                                    │
│  trans_delay = pkt_size × Σ(1/capacity)  ∈ ℝ^P                                   │
│  delay = queue_delay + trans_delay  ∈ ℝ^P                                        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 训练结果与参数更新机制

### 9.1 训练结果文件结构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         checkpoint 文件夹结构                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

ckpt_dir_128/
├── checkpoint                           # 检查点元数据文件
├── 01-0.82.data-00000-of-00001        # 第1轮权重文件 (val_loss=0.82)
├── 01-0.82.index                       # 第1轮索引文件
├── 02-0.59.data-00000-of-00001        # 第2轮权重文件 (val_loss=0.59) ← 最佳
├── 02-0.59.index                       # 第2轮索引文件
├── 03-2.38.data-00000-of-00001        # 第3轮权重文件 (val_loss=2.38)
├── ...
├── 15-0.76.data-00000-of-00001        # 第15轮权重文件 (val_loss=0.76)
└── checkpoint                          # 元数据: 记录最佳模型路径

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         文件命名规则: {epoch}-{val_loss}                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

示例: 02-0.59
• 02 = 第2个epoch
• 0.59 = 验证集上的MAPE (Mean Absolute Percentage Error)

checkpoint 元数据内容:
model_checkpoint_path: "02-0.59"           ← 指向最佳模型
all_model_checkpoint_paths: "02-0.59"      ← 所有保存的模型列表

作用:
• 记录哪个epoch的模型是最优的
• predict.py 通过读取此文件找到最佳checkpoint
```

### 9.2 checkpoint 保存的权重内容

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         checkpoint 包含的权重详解                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

ckpt_dir_128/02-0.59 文件包含 RouteNet_Fermi 模型的所有可训练参数:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   1. 嵌入层权重 (Embedding Networks)                                       │
│   ─────────────────────────────────────────                                │
│                                                                             │
│   path_embedding:                                                           │
│   • kernel (17 → 32):  17维流量特征 → 32维隐藏空间                         │
│   • 包含: traffic, packets, time_dist参数等                                │
│                                                                             │
│   link_embedding:                                                          │
│   • kernel (5 → 32):   5维链路特征 → 32维隐藏空间                         │
│   • 包含: capacity, load, scheduling_policy等                              │
│                                                                             │
│   queue_embedding:                                                         │
│   • kernel (5 → 32):   5维队列特征 → 32维隐藏空间                         │
│   • 包含: queue_size, priority, weight等                                   │
│                                                                             │
│   2. GRU单元权重 (Message Passing)                                         │
│   ──────────────────────────────────────                                   │
│                                                                             │
│   path_update (GRU):                                                       │
│   • reset_gate: (64 → 32)                                                 │
│   • update_gate: (64 → 32)                                                │
│   • candidate: (64 → 32)                                                  │
│   • 处理路径跳序列                                                         │
│                                                                             │
│   link_update (GRU):                                                      │
│   • 聚合队列信息到链路状态                                                │
│                                                                             │
│   queue_update (GRU):                                                     │
│   • 聚合路径信息到队列状态                                                │
│                                                                             │
│   3. Readout层权重 (Prediction)                                           │
│   ────────────────────────────                                             │
│                                                                             │
│   readout_path:                                                             │
│   • dense_1: (16 → 16)                                                    │
│   • dense_2: (16 → 16)                                                    │
│   • dense_3: (16 → 1)  → 输出occupancy                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 模型训练过程学到的内容

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         训练过程中各组件学习的内容                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   【组件1】Path嵌入网络 (17→32维)                                         │
│   ─────────────────────────────────────                                   │
│                                                                             │
│   输入特征 (17维):                                                         │
│   • traffic: 平均流量                                                     │
│   • packets: 包数量                                                        │
│   • model: 流量模型 (one-hot, 6维)                                        │
│   • eq_lambda: 等效到达率                                                 │
│   • avg_t_on/off: On-Off参数                                             │
│   • ar_a, sigma: 自回归参数                                                │
│   • exp_max_factor: 指数最大因子                                          │
│   • pkts_lambda_on: 开状态到达率                                          │
│   • avg_pkts_lambda: 平均包到达率                                          │
│                                                                             │
│   学习目标:                                                                │
│   • 相似的流量模式 → 相似的向量表示                                       │
│   • 不同流量模型 (EXP vs ONOFF vs PPBP) → 不同嵌入                       │
│   • 流量突发性与嵌入表示的关系                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   【组件2】Link嵌入网络 (5→32维)                                          │
│   ─────────────────────────────────────                                   │
│                                                                             │
│   输入特征 (5维):                                                          │
│   • load: 链路负载 (流量/容量)                                            │
│   • scheduling_policy: 调度策略 (one-hot, 4维)                           │
│                                                                             │
│   学习目标:                                                                │
│   • 负载水平 → 拥塞程度编码                                               │
│   • 调度策略 → 对排队的影响                                               │
│   • 负载×策略 → 交互效应                                                 │
│                                                                             │
│   场景映射:                                                               │
│   • 低负载 + FIFO → 正常服务状态                                         │
│   • 高负载 + FIFO → 拥塞状态                                             │
│   • 高负载 + WFQ → 公平调度状态                                         │
│   • 高负载 + SP → 优先级抢占状态                                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   【组件3】Queue嵌入网络 (5→32维)                                         │
│   ─────────────────────────────────────                                   │
│                                                                             │
│   输入特征 (5维):                                                          │
│   • queue_size: 队列大小                                                  │
│   • priority: 优先级 (one-hot, 3维)                                       │
│   • weight: 权重 (WFQ/DRR)                                               │
│                                                                             │
│   学习目标:                                                                │
│   • 队列容量约束 → 溢出阈值                                              │
│   • 优先级关系 → 高优先级抢占效应                                         │
│   • 权重公平性 → WFQ的服务分配                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   【组件4】GRU消息传递权重                                                │
│   ─────────────────────────────────                                      │
│                                                                             │
│   路径GRU (path_update):                                                  │
│   • reset_gate: 决定丢弃多少之前跳的信息                                  │
│   • update_gate: 决定保留多少之前的状态                                  │
│   • candidate: 融合当前跳的 (queue, link) 信息                           │
│                                                                             │
│   队列GRU (queue_update):                                                │
│   • 竞争流的数量和强度如何影响队列占用                                    │
│   • 不同流量的聚合效应 (非线性的!)                                       │
│   • 调度策略如何影响队列服务                                              │
│                                                                             │
│   链路GRU (link_update):                                                 │
│   • 多队列共享链路的公平性                                                │
│   • 负载分布                                                                │
│   • 潜在瓶颈位置                                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   【组件5】Readout网络 (16→16→1)                                         │
│   ──────────────────────────────────────                                   │
│                                                                             │
│   功能: occupancy = Readout(path_state_at_this_hop)                         │
│                                                                             │
│   学习目标:                                                                │
│   • 高流量 + 高竞争 → 高occupancy                                        │
│   • 低流量 + 低竞争 → 低occupancy                                        │
│   • 突发流量 → 瞬时高occupancy                                           │
│                                                                             │
│   物理意义:                                                                │
│   Readout实际上学习了一个"排队模型":                                     │
│   occupancy = f(path_state, queue_state)                                   │
│   而不是直接用公式: occupancy = λ / (μ - λ) (M/M/1假设)                │
│   这使得模型可以学习复杂的非线性排队行为                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 训练过程的物理意义

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         训练过程的本质                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   RouteNet通过训练习得的"知识"可以分为三层:                                 │
│                                                                             │
│   Layer 1: 特征表示 (嵌入层)                                               │
│   ────────────────────────────────────────                                  │
│   • 流量特征的连续化表示                                                   │
│   • 链路负载的归一化表示                                                   │
│   • 队列优先级的语义编码                                                   │
│   原始输入 → 有意义的向量空间                                             │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│   Layer 2: 物理关系 (消息传递)                                             │
│   ────────────────────────────────────────                                  │
│   • 流量如何汇聚到链路                                                    │
│   • 链路负载如何影响队列服务                                               │
│   • 队列状态如何决定路径延迟                                              │
│   • 路径信息如何反馈影响队列和链路                                        │
│   F ↔ Q ↔ L 的循环依赖关系                                               │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│   Layer 3: 预测映射 (Readout)                                              │
│   ──────────────────────────────────                                        │
│   • 路径状态 → occupancy                                                  │
│   • occupancy + capacity → delay                                          │
│   • delay累加 → 端到端延迟                                                │
│   隐藏状态 → 可解释的物理量                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.5 训练时的参数更新机制

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         一次完整训练迭代的流程                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   以 main.py 中的训练配置为例:                                              │
│   model.fit(                                                                │
│       ds_train,                                                             │
│       epochs=5,                                                             │
│       steps_per_epoch=200,  ← 每个epoch 200步                             │
│       validation_data=ds_validation,                                       │
│       ...                                                                   │
│   )                                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

一次完整的参数更新 (Forward + Backward + Update):

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Step 1: Forward Pass (前向传播)                                          │
│   ───────────────────────────────────────                                   │
│   输入数据 ──► 嵌入层 ──► 消息传递(x8) ──► Readout ──► 预测                │
│                    ↓            ↓              ↓                           │
│                 权重W1       权重W2        权重W3                          │
│   此时: 所有层的权重保持不变                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                                    ↓ 计算 Loss = f(预测, 真实值)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Step 2: Backward Pass (反向传播)                                          │
│   ───────────────────────────────────────                                   │
│   计算每个权重对Loss的梯度:                                                  │
│   ∂Loss/∂W3 ──► ∂Loss/∂W2 ──► ∂Loss/∂W1                                 │
│   此时: 所有层的权重仍然保持不变                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                                    ↓ 得到梯度 g1, g2, g3
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Step 3: Parameter Update (参数更新)                                      │
│   ──────────────────────────────────────                                    │
│   W1 = W1 - learning_rate × g1                                            │
│   W2 = W2 - learning_rate × g2                                            │
│   W3 = W3 - learning_rate × g3                                            │
│   此时: 所有权重同时更新！                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                            ========= 一轮参数更新完成 =========
```

### 9.6 完整的训练循环结构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         完整的训练循环结构                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   epochs = 5, steps_per_epoch = 200                                        │
│                                                                             │
│   for epoch in range(5):                                                   │
│       for step in range(200):                                              │
│           ┌─────────────────────────────────────────────────────┐          │
│           │  ① Forward:  完整前向传播 (嵌入+消息传递+Readout) │          │
│           │  ② Loss:     计算 MAPE = |预测 - 真实| / 真实     │          │
│           │  ③ Backward: 反向传播计算所有参数的梯度            │          │
│           │  ④ Update:   W = W - lr × grad                     │          │
│           │  ─────────── 一次参数更新完成 ───────────          │          │
│           └─────────────────────────────────────────────────────┘          │
│       Validation: 在验证集上评估模型 (不更新参数，只计算指标)              │
│       Save checkpoint: epoch-{val_loss}                                    │
│                                                                             │
│   总参数更新次数: 5 × 200 = 1000 次                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.7 端到端训练的核心原理

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         为什么必须等所有过程完成才更新？                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Q: 为什么不是每层执行完就更新？                                            │
│                                                                             │
│   A: 这叫做 "End-to-End Training" (端到端训练)                                │
│                                                                             │
│   端到端训练:                                                               │
│                                                                             │
│   嵌入层 ──► 消息传递 ──► Readout ──► Loss                                 │
│       ↓              ↓            ↓         ↑                             │
│      W1             W2           W3         │                             │
│                                            │                             │
│   Loss对W1的梯度 = f(Loss对W3的梯度, W3的结构, W2的结构)                 │
│                                                                             │
│   关键点:                                                                   │
│   • W1的更新需要知道Loss对W1的梯度                                        │
│   • 这个梯度需要从后向前计算                                               │
│   • 必须先计算W3, W2的梯度，才能算W1的梯度                                │
│                                                                             │
│   所以: 必须等所有前向传播完成，才能统一反向传播，统一更新！                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.8 测试/推理时的执行流程

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         测试时是否只过预测公式？                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   答案: 不是！测试时执行完整的forward pass！                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   训练时 vs 测试时                                                  │   │
│   │   ─────────────────                                                │   │
│   │                                                                     │   │
│   │   训练时:                                                           │   │
│   │   输入 → 嵌入 → 消息传递(×8) → Readout → 预测延迟          │   │
│   │           ↓                                                │   │
│   │     计算loss → 反向传播 → 更新权重                             │   │
│   │                                                                     │   │
│   │   测试/推理时:                                                      │   │
│   │   输入 → 嵌入 → 消息传递(×8) → Readout → 预测延迟          │   │
│   │           ↓                                                │   │
│   │     【不更新权重】← 权重来自checkpoint                        │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   call() 函数的完整执行步骤 (delay_model.py 第64-162行):                    │
│                                                                             │
│   Step 1: 输入解析 (第65-89行)                                              │
│   ─────────────────────────────────                                        │
│   inputs = {traffic, packets, capacity, queue_to_path, link_to_path, ...}  │
│                                                                             │
│   Step 2: 预处理 (第91-94行)                                               │
│   ──────────────────────                                                   │
│   load = Σ(path_traffic) / capacity  ← 有公式                              │
│   pkt_size = traffic / packets         ← 有公式                             │
│                                                                             │
│   Step 3: 嵌入层 (第97-116行) ← 【神经网络，训练学习的】                   │
│   ────────────────────────────                                             │
│   path_state = path_embedding(input_features)  ← 使用checkpoint权重       │
│   link_state = link_embedding([load, policy])     ← 使用checkpoint权重      │
│   queue_state = queue_embedding([queue_size, ...]) ← 使用checkpoint权重    │
│                                                                             │
│   Step 4: 消息传递 × 8次迭代 (第119-149行) ← 【神经网络，训练学习的】      │
│   ────────────────────────────────────────────────────────                 │
│   for it in range(8):                                                      │
│       queue_gather = gather(queue_state, ...)  ← 使用checkpoint权重       │
│       path_state_sequence, path_state = GRU(...) ← 使用checkpoint权重     │
│       queue_state = GRU(...)                       ← 使用checkpoint权重  │
│       link_state = GRU(...)                       ← 使用checkpoint权重  │
│                                                                             │
│   Step 5: Readout预测 (第151-156行) ← 【神经网络，训练学习的】              │
│   ─────────────────────────────────                                        │
│   occupancy = readout_path(path_state_sequence)  ← 使用checkpoint权重     │
│                                                                             │
│   Step 6: 延迟计算 (第158-162行) ← 【纯公式，无权重】                      │
│   ─────────────────────────────────                                        │
│   queue_delay = Σ(occupancy / capacity)  ← 公式                            │
│   trans_delay = pkt_size × Σ(1 / capacity)  ← 公式                         │
│   return queue_delay + trans_delay                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.9 测试时的计算类型区分

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         测试时的"有公式" vs "神经网络"部分                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   【有公式的部分】测试时仍然计算，但无需训练                                 │
│   ─────────────────────────────────────────                                 │
│   • 负载计算: load = Σ(traffic) / capacity                                  │
│   • 传输延迟: trans_delay = pkt_size × Σ(1/capacity)                         │
│   • 排队延迟公式部分: queue_delay = Σ(occupancy / capacity)                 │
│                                                                             │
│   【神经网络的部分】测试时使用训练好的权重                                    │
│   ───────────────────────────────────────────                               │
│   • 嵌入层 (Embedding): 权重来自checkpoint                                  │
│   • GRU消息传递(×8): 权重来自checkpoint，推理8次迭代                       │
│   • Readout层: 权重来自checkpoint                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   测试时的计算量分布 (估算):                                                │
│                                                                             │
│   计算类型          │  计算量      │  占比                                  │
│   ─────────────────┼────────────┼───────────                             │
│   嵌入层           │   小        │   ~5%                                 │
│   GRU消息传递(×8) │   大        │   ~80%                                │
│   Readout         │   中        │   ~10%                                │
│   延迟公式         │   很小      │   ~5%                                 │
│                                                                             │
│   结论: 测试时80%的计算在GRU消息传递！                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.10 训练结果的核心含义

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         checkpoint 文件保存的真正含义                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   checkpoint文件保存的是:                                                     │
│                                                                             │
│   一个"网络排队行为的神经网络近似器"                                        │
│                                                                             │
│   输入: 流量 + 拓扑 + 路由 + 调度                                            │
│        │                                                                    │
│        ▼                                                                    │
│   通过嵌入层 → 连续向量空间                                                  │
│        │                                                                    │
│        ▼                                                                    │
│   通过消息传递 → 状态迭代更新                                               │
│        │                                                                    │
│        ▼                                                                    │
│   通过Readout → occupancy预测                                               │
│        │                                                                    │
│        ▼                                                                    │
│   计算延迟 = Σ(occupancy / capacity) + Σ(trans_delay)                        │
│        │                                                                    │
│        ▼                                                                    │
│   输出: 预测延迟                                                            │
│                                                                             │
│   训练目标: 调整权重，使得预测延迟 ≈ 真实延迟                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   模型最终"知道"什么:                                                       │
│   ✓ 知道流量特征如何影响排队                                               │
│   ✓ 知道链路负载如何与流量交互                                             │
│   ✓ 知道调度策略如何影响服务公平性                                         │
│   ✓ 知道路径上各跳的延迟如何累积                                           │
│   ✓ 知道不同规模网络的相似模式                                             │
│                                                                             │
│   模型"不知道"什么:                                                         │
│   ✗ 不知道全新的流量模型                                                   │
│   ✗ 不知道完全不同的网络拓扑结构                                           │
│   ✗ 不能给出可解释的物理公式                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   训练结果的泛化能力来源:                                                   │
│                                                                             │
│   ✓ 关系矩阵提供了拓扑结构的不变性                                         │
│     • 不是学习特定节点的属性，而是学习节点间的关系                         │
│                                                                             │
│   ✓ 消息传递提供了尺度无关的归纳偏置                                        │
│     • 同样的 GRU 权重可以处理不同长度的路径                               │
│     • 同样的聚合逻辑可以处理不同数量的竞争流                               │
│                                                                             │
│   ✓ 嵌入层提供了流量的连续表示                                             │
│     • 训练时见过的流量模式 → 插值                                          │
│     • 训练时未见过的流量参数 → 外推 (可能不准)                            │
│                                                                             │
│   ✗ 但无法泛化:                                                           │
│     • 新的流量模型类型 (论文明确承认)                                     │
│     • 完全不同的网络拓扑类型                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录: 完整代码对照表

| 模块 | 文件 | 核心函数 |
|------|------|----------|
| 数据API | datanetAPI.py | `DatanetAPI`, `Sample` |
| 超图构建 | data_generator.py | `network_to_hypergraph` |
| 关系矩阵 | data_generator.py | `hypergraph_to_input_data` |
| 模型定义 | delay_model.py | `RouteNet_Fermi` |
| 嵌入层 | delay_model.py | `path_embedding`, `link_embedding`, `queue_embedding` |
| 消息传递 | delay_model.py | `path_update`, `link_update`, `queue_update` |
| 预测层 | delay_model.py | `readout_path` |
| 训练入口 | main.py | `model.fit()` |
| 预测推理 | predict.py | `model.predict()` |

---

*报告生成时间: 2026-05-07*
*最后更新: 新增第9章「训练结果与参数更新机制」*
