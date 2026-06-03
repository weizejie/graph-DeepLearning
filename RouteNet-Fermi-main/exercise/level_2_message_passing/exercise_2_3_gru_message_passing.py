"""
Exercise 2.3: GRU 消息传递

目标：理解 RouteNet-Fermi 中使用的 GRU 消息传递机制

RouteNet-Fermi 使用 GRU (Gated Recurrent Unit) 作为消息更新函数：
- GRU 可以记住之前的状态
- 适合迭代消息传递场景
- 比简单的神经网络有更好的梯度流动

这与在序列数据上使用 RNN 类似，只不过是在图结构上进行消息传递！
"""

import torch
import torch.nn as nn


def create_simple_topology():
    """
    创建简单的网络拓扑
    
    场景：3个路径(P)、2个链路(L)、3个队列(Q)
    
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
    """
    
    num_paths = 3
    num_links = 2
    num_queues = 3
    
    # 邻接关系（使用列表形式的索引）
    link_to_path = [[0, 1, 2], []]
    queue_to_path = [[0], [1], [2]]
    path_to_queue = [[0], [1], [2]]
    queue_to_link = [[0], [0], [0]]
    path_to_link = [[0], [0], [0]]
    
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


def create_initial_states(topology):
    """创建初始节点状态"""
    torch.manual_seed(42)
    
    path_state_dim = 8
    link_state_dim = 8
    queue_state_dim = 8
    
    path_state = torch.randn(topology['num_paths'], path_state_dim)
    link_state = torch.randn(topology['num_links'], link_state_dim)
    queue_state = torch.randn(topology['num_queues'], queue_state_dim)
    
    return path_state, link_state, queue_state


class GRUMessagePassing(nn.Module):
    """
    基于 GRU 的消息传递
    
    这是 RouteNet-Fermi 使用的核心机制！
    """
    
    def __init__(self, state_dim):
        super().__init__()
        
        self.state_dim = state_dim
        
        # ========== TODO 1: 定义 GRU 单元 ==========
        # GRU 可以自动处理状态更新
        # 输入维度 = 消息维度 (聚合后的邻居信息)
        # 输出维度 = 状态维度
        # 
        # self.gru = nn.GRUCell(...)
        self.gru = nn.GRUCell(state_dim, state_dim)
        # ============================================
    
    def aggregate_neighbors(self, node_state, neighbors):
        """
        聚合邻居节点的状态
        
        简单的求和聚合
        """
        num_nodes = node_state.shape[0]
        aggregated = []
        
        for i in range(num_nodes):
            neighbor_ids = neighbors[i]
            
            if len(neighbor_ids) > 0:
                # ========== TODO 2: 获取并聚合邻居状态 ==========
                # 
                # neighbor_states = ...
                # agg = ...
                neighbor_states = node_state[neighbor_ids]
                agg = neighbor_states.sum(dim=0)
                # ================================================
            else:
                # 无邻居时，使用零向量
                agg = torch.zeros(self.state_dim)
            
            aggregated.append(agg)
        
        return torch.stack(aggregated)
    
    def forward(self, node_state, neighbors):
        """
        GRU 消息传递前向传播
        
        参数:
            node_state: 当前节点状态 [num_nodes, state_dim]
            neighbors: 邻接表
        
        返回:
            updated_state: 更新后的节点状态 [num_nodes, state_dim]
        """
        # ========== TODO 3: 聚合邻居消息 ==========
        # 
        # neighbor_agg = self.aggregate_neighbors(...)
        neighbor_agg = self.aggregate_neighbors(node_state, neighbors)
        # ==========================================
        
        # ========== TODO 4: 使用 GRU 更新状态 ==========
        # GRU 的前向传播:
        # hidden_state = gru(input, hidden_state)
        # 
        # updated_state = self.gru(...)
        updated_state = self.gru(neighbor_agg, node_state)
        # =============================================
        
        return updated_state


def simulate_message_passing_iterations(topology, path_state, link_state, queue_state, 
                                         path_gru, link_gru, queue_gru, num_iterations=3):
    """
    模拟 RouteNet-Fermi 的多次消息传递迭代
    
    每次迭代:
    1. Link/Queue → Path: 更新路径状态
    2. Path → Queue: 更新队列状态
    3. Queue → Link: 更新链路状态
    """
    
    print("\n" + "=" * 60)
    print("消息传递迭代")
    print("=" * 60)
    
    for iteration in range(num_iterations):
        print(f"\n--- 迭代 {iteration + 1} ---")
        
        # ========== TODO 5: 消息传递步骤 ==========
        
        # 步骤1: Link + Queue → Path
        # 聚合链路和队列信息，更新路径状态
        # 
        # 1.1: 从 Queue 聚合到 Path
        # queue_to_path_agg = queue_gru(queue_state, topology['queue_to_path'])
        queue_to_path_agg = queue_gru.aggregate_neighbors(queue_state, topology['queue_to_path'])
        
        # 1.2: 从 Link 聚合到 Path  
        link_to_path_agg = link_gru.aggregate_neighbors(link_state, topology['link_to_path'])
        
        # 1.3: 合并 Queue 和 Link 信息
        combined_path_input = queue_to_path_agg + link_to_path_agg
        
        # 1.4: 使用 GRU 更新路径状态
        # path_state = path_gru(combined_path_input, path_state)
        path_state = path_gru(combined_path_input, path_state)
        
        # 步骤2: Path → Queue
        # 
        # path_to_queue_agg = ...
        # queue_state = queue_gru(path_to_queue_agg, queue_state)
        path_to_queue_agg = path_gru.aggregate_neighbors(path_state, topology['path_to_queue'])
        queue_state = queue_gru(path_to_queue_agg, queue_state)
        
        # 步骤3: Queue → Link
        # 
        # queue_to_link_agg = ...
        # link_state = link_gru(queue_to_link_agg, link_state)
        queue_to_link_agg = queue_gru.aggregate_neighbors(queue_state, topology['queue_to_link'])
        link_state = link_gru(queue_to_link_agg, link_state)
        
        # ==========================================
        
        # 打印每种节点状态的统计信息
        print(f"  Path 状态: mean={path_state.mean():.4f}, std={path_state.std():.4f}")
        print(f"  Link 状态: mean={link_state.mean():.4f}, std={link_state.std():.4f}")
        print(f"  Queue 状态: mean={queue_state.mean():.4f}, std={queue_state.std():.4f}")
    
    return path_state, link_state, queue_state


def main():
    print("=" * 60)
    print("Exercise 2.3: GRU 消息传递")
    print("=" * 60)
    
    # 创建拓扑
    topology = create_simple_topology()
    
    # 创建初始状态
    path_state, link_state, queue_state = create_initial_states(topology)
    
    print(f"初始状态形状:")
    print(f"  Path: {path_state.shape}")
    print(f"  Link: {link_state.shape}")
    print(f"  Queue: {queue_state.shape}")
    
    # 创建 GRU 消息传递层
    path_state_dim = path_state.shape[1]
    link_state_dim = link_state.shape[1]
    queue_state_dim = queue_state.shape[1]
    
    path_gru = GRUMessagePassing(path_state_dim)
    link_gru = GRUMessagePassing(link_state_dim)
    queue_gru = GRUMessagePassing(queue_state_dim)
    
    # 模拟多次迭代
    final_path_state, final_link_state, final_queue_state = simulate_message_passing_iterations(
        topology, path_state, link_state, queue_state,
        path_gru, link_gru, queue_gru,
        num_iterations=3
    )
    
    print("\n" + "=" * 60)
    print("练习完成！你已经理解了 RouteNet-Fermi 的核心机制！")
    print("=" * 60)
    print("""
提示: RouteNet-Fermi 的核心思想:
1. 将网络建模为 Path、Link、Queue 三种节点的图
2. 使用 GRU 进行消息传递和状态更新
3. 迭代多次让信息在网络中传播
4. 最终用路径状态预测性能指标（如延迟）
""")


if __name__ == "__main__":
    main()
