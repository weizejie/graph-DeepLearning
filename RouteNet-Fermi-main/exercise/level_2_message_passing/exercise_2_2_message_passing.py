"""
Exercise 2.2: 消息传递机制

目标：理解图神经网络中的消息传递（Message Passing）机制

消息传递的核心步骤：
1. 收集（Gather）：从邻居节点收集消息
2. 聚合（Aggregate）：将多个邻居的消息合并
3. 更新（Update）：使用聚合的消息更新节点状态

这与 RouteNet-Fermi 中的迭代消息传递完全一致！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_simple_data():
    """创建简单的测试数据"""
    torch.manual_seed(42)
    
    # 3个节点，每个节点8维特征
    num_nodes = 3
    state_dim = 8
    
    # 节点状态
    node_state = torch.randn(num_nodes, state_dim)
    
    # 邻接表：每个节点的邻居
    # adj_list[0] = [1, 2] 表示节点0与节点1、2相连
    adj_list = [
        [1, 2],  # 节点0的邻居
        [0, 2],  # 节点1的邻居
        [0, 1],  # 节点2的邻居
    ]
    
    return node_state, adj_list


class MessagePassingLayer(nn.Module):
    """基础的消息传递层"""
    
    def __init__(self, state_dim, message_dim=None):
        super().__init__()
        
        if message_dim is None:
            message_dim = state_dim
            
        self.state_dim = state_dim
        self.message_dim = message_dim
        
        # ========== TODO 1: 定义消息神经网络 ==========
        # 将节点状态转换为消息
        # 
        # self.message_net = nn.Sequential(...)
        self.message_net = nn.Sequential(
            nn.Linear(state_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        # ============================================
        
        # ========== TODO 2: 定义更新神经网络 ==========
        # 将当前状态和聚合消息结合，更新节点状态
        # 
        # self.update_net = nn.Sequential(...)
        self.update_net = nn.Sequential(
            nn.Linear(state_dim + message_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        # ============================================
    
    def gather_messages(self, node_state, neighbors):
        """
        收集邻居节点的消息
        
        参数:
            node_state: 所有节点的状态 [num_nodes, state_dim]
            neighbors: 邻接表 (list of lists)
        
        返回:
            messages: 每个节点的消息 [num_nodes, message_dim]
        """
        num_nodes = node_state.shape[0]
        messages = []
        
        for i in range(num_nodes):
            # ========== TODO 2.1: 获取邻居节点状态 ==========
            # 
            # neighbor_states = node_state[neighbors[i]]
            neighbor_states = node_state[neighbors[i]]
            # ================================================
            
            # ========== TODO 2.2: 为每个邻居生成消息 ==========
            # 
            # neighbor_messages = ...
            neighbor_messages = self.message_net(neighbor_states)
            # ================================================
            
            # ========== TODO 2.3: 聚合邻居消息 (求和) ==========
            # 
            # aggregated = ...
            aggregated = neighbor_messages.sum(dim=0)
            # ================================================
            
            messages.append(aggregated)
        
        # 堆叠成 [num_nodes, message_dim]
        return torch.stack(messages)
    
    def forward(self, node_state, neighbors):
        """
        完整的前向传播
        
        参数:
            node_state: 当前节点状态 [num_nodes, state_dim]
            neighbors: 邻接表
        
        返回:
            updated_state: 更新后的节点状态 [num_nodes, state_dim]
        """
        # ========== TODO 3: 实现完整消息传递流程 ==========
        # 
        # 1. 收集邻居消息
        # 2. 聚合消息
        # 3. 与当前状态拼接
        # 4. 更新状态
        # 
        # messages = self.gather_messages(...)
        # concatenated = torch.cat([...], dim=...)
        # updated_state = self.update_net(...)
        
        messages = self.gather_messages(node_state, neighbors)
        concatenated = torch.cat([node_state, messages], dim=-1)
        updated_state = self.update_net(concatenated)
        # ===================================================
        
        return updated_state


def aggregate_messages_manual(node_state, neighbors):
    """
    手动实现聚合（不使用神经网络）
    用于理解聚合的概念
    """
    print("\n" + "=" * 60)
    print("手动聚合示例")
    print("=" * 60)
    
    num_nodes = node_state.shape[0]
    
    for i in range(num_nodes):
        neighbor_ids = neighbors[i]
        print(f"\n节点 {i}:")
        print(f"  邻居节点: {neighbor_ids}")
        
        # 获取邻居状态
        neighbor_states = node_state[neighbor_ids]
        print(f"  邻居状态形状: {neighbor_states.shape}")
        
        # ========== TODO 4: 实现不同的聚合方式 ==========
        
        # 方式1: 求和 (Sum)
        # sum_agg = ...
        sum_agg = neighbor_states.sum(dim=0)
        
        # 方式2: 平均 (Mean)
        # mean_agg = ...
        mean_agg = neighbor_states.mean(dim=0)
        
        # 方式3: 最大值 (Max)
        # max_agg = ...
        max_agg = neighbor_states.max(dim=0)[0]
        
        # ================================================
        
        print(f"  求和聚合: {sum_agg[:3]}...")  # 只打印前3维
        print(f"  平均聚合: {mean_agg[:3]}...")
        print(f"  最大聚合: {max_agg[:3]}...")


def test_message_passing():
    """测试消息传递层"""
    print("\n" + "=" * 60)
    print("测试消息传递层")
    print("=" * 60)
    
    # 创建数据
    node_state, adj_list = create_simple_data()
    
    print(f"输入节点状态形状: {node_state.shape}")
    print(f"邻接表: {adj_list}")
    
    # 创建消息传递层
    state_dim = 8
    mp_layer = MessagePassingLayer(state_dim)
    
    # 前向传播
    updated_state = mp_layer(node_state, adj_list)
    
    print(f"\n输出节点状态形状: {updated_state.shape}")
    print(f"\n原始状态 (节点0): {node_state[0, :3]}")
    print(f"更新状态 (节点0): {updated_state[0, :3]}")


def main():
    print("=" * 60)
    print("Exercise 2.2: 消息传递机制")
    print("=" * 60)
    
    # 创建数据
    node_state, adj_list = create_simple_data()
    
    print(f"节点状态形状: {node_state.shape}")
    print(f"邻接表: {adj_list}")
    
    # 手动聚合示例
    aggregate_messages_manual(node_state, adj_list)
    
    # 测试消息传递层
    test_message_passing()
    
    print("\n" + "=" * 60)
    print("练习完成！消息传递是图神经网络的核心操作")
    print("=" * 60)


if __name__ == "__main__":
    main()
