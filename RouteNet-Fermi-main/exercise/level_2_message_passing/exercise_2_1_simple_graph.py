"""
Exercise 2.1: 简单的图结构表示

目标：理解 RouteNet-Fermi 中的图结构表示方法

在 RouteNet-Fermi 中，网络拓扑被表示为三种节点之间的邻接关系：
- Path（路径）：数据流经过的路径
- Link（链路）：物理网络链路
- Queue（队列）：链路上的排队缓冲区

本练习使用简单的合成数据来理解这些关系如何表示。
"""

import torch


def create_simple_topology():
    """
    创建简单的网络拓扑
    
    场景：3个路径，2个链路，3个队列
    
    拓扑结构：
    
              Link 0
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
     Queue 0  Queue 1  Queue 2
        │        │        │
        ▼        ▼        ▼
      Path 0   Path 1   Path 2
    
    关系说明：
    - Link 0 连接到所有 3 个队列
    - 每个 Queue 连接到对应的 Path
    """
    
    # 节点数量
    num_paths = 3
    num_links = 2
    num_queues = 3
    
    # ========== TODO 1: 定义邻接关系 ==========
    # 
    # link_to_path[i] = [path_id_1, path_id_2, ...]
    # 表示 Link i 连接了哪些 Path
    # 
    # 提示：根据上面的拓扑图填写
    # 
    # link_to_path = [[...], [...]]  # Link 0 连接所有路径，Link 1 无连接
    link_to_path = [[0, 1, 2], []]
    
    # queue_to_path[i] = [path_id_1, path_id_2, ...]
    # 表示 Queue i 连接到哪些 Path
    # 
    # queue_to_path = [[...], [...], [...]]
    queue_to_path = [[0], [1], [2]]
    
    # path_to_queue[i] = [queue_id_1, queue_id_2, ...]
    # 表示 Path i 经过哪些 Queue
    # 
    # path_to_queue = [[...], [...], [...]]
    path_to_queue = [[0], [1], [2]]
    
    # queue_to_link[i] = [link_id_1, link_id_2, ...]
    # 表示 Queue i 位于哪些 Link 上
    # 
    # queue_to_link = [[...], [...], [...]]
    queue_to_link = [[0], [0], [0]]
    
    # path_to_link[i] = [link_id_1, link_id_2, ...]
    # 表示 Path i 经过哪些 Link
    # 
    # path_to_link = [[...], [...], [...]]
    path_to_link = [[0], [0], [0]]
    
    # ==========================================
    
    return {
        'num_paths': num_paths,
        'num_links': num_links,
        'num_queues': num_queues,
        'link_to_path': link_to_path,
        'queue_to_path': queue_to_path,
        'path_to_queue': path_to_queue,
        'queue_to_link': queue_to_link,
        'path_to_link': path_to_link,
    }


def create_node_features(topology):
    """
    为每个节点创建特征向量
    
    在真实场景中：
    - Path 节点特征：流量特征（traffic, packets, model等）
    - Link 节点特征：链路容量、调度策略
    - Queue 节点特征：队列大小、优先级、权重
    """
    num_paths = topology['num_paths']
    num_links = topology['num_links']
    num_queues = topology['num_queues']
    
    path_state_dim = 8
    link_state_dim = 8
    queue_state_dim = 8
    
    torch.manual_seed(42)
    
    # 路径节点特征
    path_state = torch.randn(num_paths, path_state_dim)
    
    # 链路节点特征
    link_state = torch.randn(num_links, link_state_dim)
    
    # 队列节点特征
    queue_state = torch.randn(num_queues, queue_state_dim)
    
    return {
        'path_state': path_state,
        'link_state': link_state,
        'queue_state': queue_state,
    }


def visualize_topology(topology):
    """可视化拓扑结构"""
    print("=" * 60)
    print("网络拓扑结构")
    print("=" * 60)
    
    print(f"\n节点数量:")
    print(f"  Paths: {topology['num_paths']}")
    print(f"  Links: {topology['num_links']}")
    print(f"  Queues: {topology['num_queues']}")
    
    print(f"\n邻接关系:")
    
    print(f"\n  Link → Path:")
    for i, paths in enumerate(topology['link_to_path']):
        print(f"    Link {i} → {paths}")
    
    print(f"\n  Queue → Path:")
    for i, paths in enumerate(topology['queue_to_path']):
        print(f"    Queue {i} → {paths}")
    
    print(f"\n  Path → Queue:")
    for i, queues in enumerate(topology['path_to_queue']):
        print(f"    Path {i} → {queues}")
    
    print(f"\n  Queue → Link:")
    for i, links in enumerate(topology['queue_to_link']):
        print(f"    Queue {i} → {links}")
    
    print(f"\n  Path → Link:")
    for i, links in enumerate(topology['path_to_link']):
        print(f"    Path {i} → {links}")


def test_gather_operations(topology, features):
    """
    测试使用 gather 操作获取邻居节点信息
    
    这是消息传递的关键操作：
    - 从邻居节点收集状态
    - 聚合后更新当前节点
    """
    print("\n" + "=" * 60)
    print("测试 Gather 操作")
    print("=" * 60)
    
    path_state = features['path_state']
    queue_state = features['queue_state']
    link_state = features['link_state']
    
    # ========== TODO 2: 实现 gather 操作 ==========
    # 
    # 假设要更新 Path 0 的状态，需要：
    # 1. 找出 Path 0 经过哪些 Queue: path_to_queue[0]
    # 2. 获取这些 Queue 的状态
    # 3. 聚合（求和/平均）得到 Path 0 的输入
    # 
    # 提示: 使用 torch.index_select 或列表推导
    
    # 获取 Path 0 经过的队列 ID
    queue_ids = topology['path_to_queue'][0]
    print(f"\nPath 0 经过的队列: {queue_ids}")
    
    # 获取这些队列的状态
    # 
    # queue_state_gathered = ...
    queue_state_gathered = queue_state[queue_ids]
    print(f"队列状态形状: {queue_state_gathered.shape}")
    
    # 聚合（求和）
    # 
    # aggregated = ...
    aggregated = queue_state_gathered.sum(dim=0)
    print(f"聚合后形状: {aggregated.shape}")
    
    # ==========================================
    
    return {
        'queue_ids': queue_ids,
        'queue_state_gathered': queue_state_gathered,
        'aggregated': aggregated,
    }


def main():
    print("=" * 60)
    print("Exercise 2.1: 简单的图结构表示")
    print("=" * 60)
    
    # 创建拓扑
    topology = create_simple_topology()
    visualize_topology(topology)
    
    # 创建节点特征
    features = create_node_features(topology)
    
    print(f"\n节点特征形状:")
    print(f"  path_state: {features['path_state'].shape}")
    print(f"  link_state: {features['link_state'].shape}")
    print(f"  queue_state: {features['queue_state'].shape}")
    
    # 测试 gather 操作
    gather_result = test_gather_operations(topology, features)
    
    print("\n" + "=" * 60)
    print("练习完成！理解图结构表示是理解 RouteNet-Fermi 的基础")
    print("=" * 60)


if __name__ == "__main__":
    main()
