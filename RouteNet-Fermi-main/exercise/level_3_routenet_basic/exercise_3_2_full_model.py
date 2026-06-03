"""
Exercise 3.2: 完整的 RouteNet-Fermi 模型

目标：理解完整的 RouteNet-Fermi 前向传播过程

完整流程：
1. 嵌入层：将原始特征转换为状态向量
2. GRU 消息传递：迭代更新节点状态
3. Readout 层：从路径状态预测延迟
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 导入上一节的嵌入层（这里简化复现）
class PathEmbedding(nn.Module):
    def __init__(self, path_state_dim=32, max_num_models=7):
        super().__init__()
        self.path_state_dim = path_state_dim
        self.max_num_models = max_num_models
        
        num_traffic_features = 10
        total_features = num_traffic_features + max_num_models
        
        self.net = nn.Sequential(
            nn.Linear(total_features, path_state_dim),
            nn.ReLU(),
            nn.Linear(path_state_dim, path_state_dim),
            nn.ReLU()
        )
    
    def forward(self, traffic_features, model):
        model_onehot = F.one_hot(model, num_classes=self.max_num_models)
        combined = torch.cat([traffic_features, model_onehot.float()], dim=-1)
        return self.net(combined)


class LinkEmbedding(nn.Module):
    def __init__(self, link_state_dim=32, num_policies=4):
        super().__init__()
        self.link_state_dim = link_state_dim
        
        self.net = nn.Sequential(
            nn.Linear(1 + num_policies, link_state_dim),
            nn.ReLU(),
            nn.Linear(link_state_dim, link_state_dim),
            nn.ReLU()
        )
    
    def forward(self, capacity, policy):
        policy_onehot = F.one_hot(policy, num_classes=4)
        capacity_unsqueezed = capacity.unsqueeze(-1)
        combined = torch.cat([capacity_unsqueezed, policy_onehot.float()], dim=-1)
        return self.net(combined)


class QueueEmbedding(nn.Module):
    def __init__(self, queue_state_dim=32, max_num_queues=3):
        super().__init__()
        self.queue_state_dim = queue_state_dim
        
        self.net = nn.Sequential(
            nn.Linear(1 + max_num_queues + 1, queue_state_dim),
            nn.ReLU(),
            nn.Linear(queue_state_dim, queue_state_dim),
            nn.ReLU()
        )
    
    def forward(self, queue_size, priority, weight):
        priority_onehot = F.one_hot(priority, num_classes=3)
        queue_size_unsqueezed = queue_size.unsqueeze(-1)
        weight_unsqueezed = weight.unsqueeze(-1)
        combined = torch.cat([queue_size_unsqueezed, priority_onehot.float(), weight_unsqueezed], dim=-1)
        return self.net(combined)


def create_sample_data():
    """创建示例数据"""
    torch.manual_seed(42)
    
    batch_size = 2
    num_paths = 3
    num_links = 2
    num_queues = 3
    
    # 流量特征
    traffic = torch.randn(batch_size, num_paths) * 1000 + 1000
    packets = torch.randn(batch_size, num_paths) * 5 + 10
    eq_lambda = torch.randn(batch_size, num_paths) * 100 + 100
    avg_pkts_lambda = torch.randn(batch_size, num_paths) * 0.5 + 1
    exp_max_factor = torch.randn(batch_size, num_paths) * 2 + 5
    pkts_lambda_on = torch.randn(batch_size, num_paths) * 0.5 + 1
    avg_t_off = torch.randn(batch_size, num_paths) * 1 + 1.5
    avg_t_on = torch.randn(batch_size, num_paths) * 1 + 1.5
    ar_a = torch.rand(batch_size, num_paths)
    sigma = torch.rand(batch_size, num_paths) * 0.5
    model = torch.randint(0, 7, (batch_size, num_paths))
    
    # 链路特征
    capacity = torch.tensor([[10000.0, 10000.0], [8000.0, 8000.0]])
    policy = torch.randint(0, 4, (batch_size, num_links))
    
    # 队列特征
    queue_size = torch.tensor([[5000.0, 5000.0, 5000.0], [4000.0, 4000.0, 4000.0]])
    priority = torch.randint(0, 3, (batch_size, num_queues))
    weight = torch.rand(batch_size, num_queues) * 3 + 1
    
    # 拓扑关系 (简化版本)
    path_to_link = [[0], [0, 1], [1]]
    link_to_path = [[0, 1], [1, 2]]
    path_to_queue = [[0], [1], [2]]
    queue_to_path = [[0], [1], [2]]
    queue_to_link = [[0], [0], [1]]
    
    return {
        'traffic': traffic, 'packets': packets, 'eq_lambda': eq_lambda,
        'avg_pkts_lambda': avg_pkts_lambda, 'exp_max_factor': exp_max_factor,
        'pkts_lambda_on': pkts_lambda_on, 'avg_t_off': avg_t_off,
        'avg_t_on': avg_t_on, 'ar_a': ar_a, 'sigma': sigma, 'model': model,
        'capacity': capacity, 'policy': policy,
        'queue_size': queue_size, 'priority': priority, 'weight': weight,
        'path_to_link': path_to_link, 'link_to_path': link_to_path,
        'path_to_queue': path_to_queue, 'queue_to_path': queue_to_path,
        'queue_to_link': queue_to_link,
    }


def prepare_traffic_features(inputs):
    """准备流量特征"""
    z_score = {
        'traffic': [1385.4, 859.8], 'packets': [1.4, 0.89],
        'eq_lambda': [1350.9, 858.3], 'avg_pkts_lambda': [0.91, 0.97],
        'exp_max_factor': [6.66, 4.71], 'pkts_lambda_on': [0.91, 1.65],
        'avg_t_off': [1.66, 2.35], 'avg_t_on': [1.66, 2.35],
        'ar_a': [0.0, 1.0], 'sigma': [0.0, 1.0],
    }
    
    traffic_list = []
    for key in ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda', 
                'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on',
                'ar_a', 'sigma']:
        x = inputs[key]
        mean, std = z_score[key]
        normalized = (x - mean) / std
        traffic_list.append(normalized)
    
    return torch.stack(traffic_list, dim=-1)


class RouteNetFermi(nn.Module):
    """
    简化版 RouteNet-Fermi 模型
    """
    
    def __init__(self, path_state_dim=32, link_state_dim=32, queue_state_dim=32,
                 iterations=8, max_num_models=7, num_policies=4, max_num_queues=3):
        super().__init__()
        
        self.path_state_dim = path_state_dim
        self.link_state_dim = link_state_dim
        self.queue_state_dim = queue_state_dim
        self.iterations = iterations
        
        # 嵌入层
        self.path_embedding = PathEmbedding(path_state_dim, max_num_models)
        self.link_embedding = LinkEmbedding(link_state_dim, num_policies)
        self.queue_embedding = QueueEmbedding(queue_state_dim, max_num_queues)
        
        # GRU 消息传递层
        self.path_gru = nn.GRUCell(link_state_dim + queue_state_dim, path_state_dim)
        self.link_gru = nn.GRUCell(queue_state_dim, link_state_dim)
        self.queue_gru = nn.GRUCell(path_state_dim, queue_state_dim)
        
        # Readout 层：从路径状态预测延迟
        # ========== TODO 1: 定义 Readout 网络 ==========
        # 
        # 输入：路径状态
        # 输出：延迟预测值
        # 
        # self.readout = nn.Sequential(...)
        self.readout = nn.Sequential(
            nn.Linear(path_state_dim, path_state_dim // 2),
            nn.ReLU(),
            nn.Linear(path_state_dim // 2, 1)
        )
        # ==============================================
    
    def aggregate(self, state, neighbors):
        """
        聚合邻居节点状态
        
        参数:
            state: 所有节点状态 [batch, num_nodes, state_dim]
            neighbors: 邻接表 (list of lists)
        
        返回:
            aggregated: 聚合后的状态 [batch, num_nodes, state_dim]
        """
        batch_size = state.shape[0]
        num_nodes = len(neighbors)
        aggregated = []
        
        for i in range(num_nodes):
            neighbor_ids = neighbors[i]
            if len(neighbor_ids) > 0:
                # ========== TODO 2: 聚合邻居状态 (求和) ==========
                # 
                # neighbor_states = state[:, neighbor_ids, :]
                # agg = neighbor_states.sum(dim=1)
                neighbor_states = state[:, neighbor_ids, :]
                agg = neighbor_states.sum(dim=1)
                # ===============================================
            else:
                agg = torch.zeros(batch_size, state.shape[2])
            aggregated.append(agg)
        
        return torch.stack(aggregated, dim=1)
    
    def forward(self, inputs):
        """
        完整的前向传播
        """
        # 1. 准备流量特征并嵌入
        traffic_features = prepare_traffic_features(inputs)
        
        # ========== TODO 3: 嵌入层 ==========
        # 
        # path_state = self.path_embedding(...)
        # link_state = self.link_embedding(...)
        # queue_state = self.queue_embedding(...)
        path_state = self.path_embedding(traffic_features, inputs['model'])
        link_state = self.link_embedding(inputs['capacity'], inputs['policy'])
        queue_state = self.queue_embedding(inputs['queue_size'], inputs['priority'], inputs['weight'])
        # ===================================
        
        # 2. GRU 消息传递迭代
        for it in range(self.iterations):
            # ========== TODO 4: 消息传递步骤 ==========
            
            # 步骤 A: Link/Queue → Path
            # 
            # 1. 聚合链路信息
            # link_agg = self.aggregate(...)
            link_agg = self.aggregate(link_state, inputs['link_to_path'])
            
            # 2. 聚合队列信息  
            # queue_agg = self.aggregate(...)
            queue_agg = self.aggregate(queue_state, inputs['queue_to_path'])
            
            # 3. 合并并更新路径状态
            # path_input = torch.cat([...], dim=-1)
            path_input = torch.cat([link_agg, queue_agg], dim=-1)
            
            # 4. GRU 更新 (需要展平 batch 维度)
            batch_size = path_state.shape[0]
            path_state_flat = path_state.view(batch_size, -1)
            path_input_flat = path_input.view(batch_size, -1)
            
            # new_path_state = self.path_gru(...)
            # path_state = new_path_state.view(batch_size, -1, self.path_state_dim)
            new_path_state = self.path_gru(path_input_flat, path_state_flat)
            path_state = new_path_state.view(batch_size, -1, self.path_state_dim)
            
            # 步骤 B: Path → Queue
            # 
            # path_to_queue_agg = self.aggregate(...)
            path_to_queue_agg = self.aggregate(path_state, inputs['path_to_queue'])
            
            # queue_state_flat = queue_state.view(...)
            # new_queue_state = self.queue_gru(...)
            # queue_state = new_queue_state.view(...)
            queue_state_flat = queue_state.view(batch_size, -1)
            path_to_queue_flat = path_to_queue_agg.view(batch_size, -1)
            new_queue_state = self.queue_gru(path_to_queue_flat, queue_state_flat)
            queue_state = new_queue_state.view(batch_size, -1, self.queue_state_dim)
            
            # 步骤 C: Queue → Link
            # 
            # queue_to_link_agg = self.aggregate(...)
            queue_to_link_agg = self.aggregate(queue_state, inputs['queue_to_link'])
            
            # link_state_flat = link_state.view(...)
            # new_link_state = self.link_gru(...)
            # link_state = new_link_state.view(...)
            link_state_flat = link_state.view(batch_size, -1)
            queue_to_link_flat = queue_to_link_agg.view(batch_size, -1)
            new_link_state = self.link_gru(queue_to_link_flat, link_state_flat)
            link_state = new_link_state.view(batch_size, -1, self.link_state_dim)
            
            # =============================================
        
        # 3. Readout 层：从路径状态预测延迟
        # ========== TODO 5: 预测延迟 ==========
        # 
        # 1. 通过 Readout 网络
        # delay_per_path = self.readout(path_state)
        delay_per_path = self.readout(path_state)
        
        # 2. 对每个样本的所有路径求和/平均，得到总延迟
        # delay = delay_per_path.sum(dim=1)  # 或 .mean(dim=1)
        delay = delay_per_path.sum(dim=1)
        # ======================================
        
        return delay


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 60)
    print("测试完整 RouteNet-Fermi 模型")
    print("=" * 60)
    
    # 创建数据
    data = create_sample_data()
    
    print(f"\n输入数据:")
    print(f"  批量大小: {data['traffic'].shape[0]}")
    print(f"  路径数: {data['traffic'].shape[1]}")
    print(f"  链路数: {data['capacity'].shape[1]}")
    print(f"  队列数: {data['queue_size'].shape[1]}")
    
    # 创建模型
    model = RouteNetFermi(
        path_state_dim=32,
        link_state_dim=32,
        queue_state_dim=32,
        iterations=8
    )
    
    # 前向传播
    output = model(data)
    
    print(f"\n输出形状: {output.shape}")
    print(f"预测延迟: {output.detach().numpy()}")
    
    print("\n" + "=" * 60)
    print("模型架构:")
    print("=" * 60)
    print(model)


def main():
    print("=" * 60)
    print("Exercise 3.2: 完整的 RouteNet-Fermi 模型")
    print("=" * 60)
    
    test_full_model()
    
    print("\n" + "=" * 60)
    print("练习完成！你已经理解了完整的 RouteNet-Fermi 模型！")
    print("=" * 60)
    print("""
核心思想总结：
1. 嵌入层：将原始网络特征转换为状态向量
2. 消息传递：迭代地更新 Path、Link、Queue 的状态
3. Readout：从最终路径状态预测性能指标

这就是 RouteNet-Fermi 能够准确预测网络性能的关键！
""")


if __name__ == "__main__":
    main()
