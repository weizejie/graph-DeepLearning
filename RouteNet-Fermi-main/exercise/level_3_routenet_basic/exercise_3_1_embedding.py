"""
Exercise 3.1: RouteNet-Fermi 嵌入层

目标：理解如何将原始网络特征转换为向量表示

在 RouteNet-Fermi 中，需要处理三种类型的输入特征：
1. Traffic (流量特征): 数值型特征，需要标准化
2. Path 特征: 包含 one-hot 编码的路由模型类型
3. Link 特征: 包含 one-hot 编码的调度策略
4. Queue 特征: 包含队列大小、优先级、权重

这些特征经过嵌入层后，转换为固定维度的向量表示。
"""

import torch
import torch.nn as nn
import numpy as np


def create_sample_inputs():
    """
    创建示例输入数据
    
    这些数据的结构与真实 DatanetAPI 数据一致
    """
    # 假设 batch_size = 2
    batch_size = 2
    
    # ========== TODO 1: 定义流量特征 (Traffic Features) ==========
    # 
    # 原始论文中的流量特征：
    # - traffic: 流量速率
    # - packets: 数据包数量
    # - eq_lambda: 等效到达率
    # - avg_pkts_lambda: 平均包到达率
    # - exp_max_factor: 指数最大因子
    # - pkts_lambda_on: 开状态包到达率
    # - avg_t_off: 平均关闭时间
    # - avg_t_on: 平均开启时间
    # - ar_a: 自回归参数 a
    # - sigma: 标准差
    
    # 模拟数值特征 (需要标准化的)
    traffic = torch.tensor([[1000.0, 500.0, 800.0], [800.0, 600.0, 700.0]])  # [batch, num_paths]
    packets = torch.tensor([[10.0, 5.0, 8.0], [8.0, 6.0, 7.0]])  # [batch, num_paths]
    eq_lambda = torch.tensor([[100.0, 50.0, 80.0], [80.0, 60.0, 70.0]])
    avg_pkts_lambda = torch.tensor([[1.0, 0.5, 0.8], [0.8, 0.6, 0.7]])
    exp_max_factor = torch.tensor([[5.0, 3.0, 4.0], [4.0, 3.5, 4.5]])
    pkts_lambda_on = torch.tensor([[1.0, 0.5, 0.8], [0.8, 0.6, 0.7]])
    avg_t_off = torch.tensor([[1.5, 1.0, 1.2], [1.2, 1.1, 1.3]])
    avg_t_on = torch.tensor([[1.5, 1.0, 1.2], [1.2, 1.1, 1.3]])
    ar_a = torch.tensor([[0.5, 0.3, 0.4], [0.4, 0.35, 0.45]])
    sigma = torch.tensor([[0.5, 0.3, 0.4], [0.4, 0.35, 0.45]])
    
    # 模拟分类特征 (需要 one-hot 编码的)
    # 路由模型类型 (max_num_models = 7)
    model = torch.tensor([[0, 1, 2], [1, 2, 0]])  # [batch, num_paths]
    
    # ===============================================================
    
    # ========== TODO 2: 定义链路特征 (Link Features) ==========
    
    # 链路容量
    capacity = torch.tensor([[10000.0, 10000.0], [8000.0, 8000.0]])  # [batch, num_links]
    
    # 调度策略 (4种: WFQ, SP, DRR, FIFO)
    policy = torch.tensor([[0, 1], [1, 2]])  # [batch, num_links]  # 策略索引
    
    # ===============================================================
    
    # ========== TODO 3: 定义队列特征 (Queue Features) ==========
    
    # 队列大小
    queue_size = torch.tensor([[5000.0, 5000.0, 5000.0], [4000.0, 4000.0, 4000.0]])  # [batch, num_queues]
    
    # 优先级 (3种: 0, 1, 2)
    priority = torch.tensor([[0, 1, 2], [1, 2, 0]])  # [batch, num_queues]
    
    # 权重 (用于 WFQ 调度)
    weight = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])  # [batch, num_queues]
    
    # ===============================================================
    
    return {
        'traffic': traffic,
        'packets': packets,
        'eq_lambda': eq_lambda,
        'avg_pkts_lambda': avg_pkts_lambda,
        'exp_max_factor': exp_max_factor,
        'pkts_lambda_on': pkts_lambda_on,
        'avg_t_off': avg_t_off,
        'avg_t_on': avg_t_on,
        'ar_a': ar_a,
        'sigma': sigma,
        'model': model,
        'capacity': capacity,
        'policy': policy,
        'queue_size': queue_size,
        'priority': priority,
        'weight': weight,
    }


class ZScoreNormalizer:
    """Z-Score 标准化器"""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, x):
        # ========== TODO 4: 实现 Z-Score 标准化 ==========
        # 
        # normalized = (x - mean) / std
        # 
        return (x - self.mean) / self.std
        # ================================================


class PathEmbedding(nn.Module):
    """
    路径嵌入层
    
    将流量特征和路径类型转换为状态向量
    """
    
    def __init__(self, path_state_dim=32, max_num_models=7):
        super().__init__()
        
        self.path_state_dim = path_state_dim
        self.max_num_models = max_num_models
        
        # 流量特征数量 (10个数值特征 + 1个分类特征)
        num_traffic_features = 10
        num_path_features = max_num_models  # one-hot 编码的 model
        
        total_features = num_traffic_features + num_path_features
        
        # ========== TODO 5: 定义嵌入网络 ==========
        # 
        # 两层全连接网络，将特征映射到 path_state_dim
        # 
        # self.net = nn.Sequential(...)
        self.net = nn.Sequential(
            nn.Linear(total_features, path_state_dim),
            nn.ReLU(),
            nn.Linear(path_state_dim, path_state_dim),
            nn.ReLU()
        )
        # ========================================
    
    def forward(self, traffic_features, model):
        """
        前向传播
        
        参数:
            traffic_features: 流量特征, shape [batch, num_paths, 10]
            model: 路由模型类型, shape [batch, num_paths]
        
        返回:
            path_state: 路径状态, shape [batch, num_paths, path_state_dim]
        """
        batch_size, num_paths, _ = traffic_features.shape
        
        # ========== TODO 6: 对 model 进行 one-hot 编码 ==========
        # 
        # model_onehot = F.one_hot(model, num_classes=self.max_num_models)
        # 形状变为 [batch, num_paths, max_num_models]
        model_onehot = torch.nn.functional.one_hot(model, num_classes=self.max_num_models)
        # =====================================================
        
        # ========== TODO 7: 拼接流量特征和模型特征 ==========
        # 
        # combined = torch.cat([...], dim=-1)
        combined = torch.cat([traffic_features, model_onehot.float()], dim=-1)
        # ====================================================
        
        # ========== TODO 8: 通过嵌入网络 ==========
        # 
        # path_state = self.net(combined)
        path_state = self.net(combined)
        # =========================================
        
        return path_state


class LinkEmbedding(nn.Module):
    """
    链路嵌入层
    
    将链路容量和调度策略转换为状态向量
    """
    
    def __init__(self, link_state_dim=32, num_policies=4):
        super().__init__()
        
        self.link_state_dim = link_state_dim
        self.num_policies = num_policies
        
        # 链路特征: 容量(1) + 策略(one-hot, 4) = 5
        num_link_features = 1 + num_policies
        
        # ========== TODO 9: 定义嵌入网络 ==========
        # 
        # self.net = nn.Sequential(...)
        self.net = nn.Sequential(
            nn.Linear(num_link_features, link_state_dim),
            nn.ReLU(),
            nn.Linear(link_state_dim, link_state_dim),
            nn.ReLU()
        )
        # ========================================
    
    def forward(self, capacity, policy):
        """
        前向传播
        
        参数:
            capacity: 链路容量, shape [batch, num_links]
            policy: 调度策略索引, shape [batch, num_links]
        
        返回:
            link_state: 链路状态, shape [batch, num_links, link_state_dim]
        """
        batch_size, num_links = capacity.shape
        
        # ========== TODO 10: 对 policy 进行 one-hot 编码 ==========
        # 
        # policy_onehot = F.one_hot(...)
        policy_onehot = torch.nn.functional.one_hot(policy, num_classes=self.num_policies)
        # ==========================================================
        
        # ========== TODO 11: 拼接容量和策略特征 ==========
        # 
        # combined = torch.cat([...], dim=-1)
        capacity_unsqueezed = capacity.unsqueeze(-1)  # [batch, num_links, 1]
        combined = torch.cat([capacity_unsqueezed, policy_onehot.float()], dim=-1)
        # ===================================================
        
        # ========== TODO 12: 通过嵌入网络 ==========
        # 
        # link_state = self.net(combined)
        link_state = self.net(combined)
        # =========================================
        
        return link_state


class QueueEmbedding(nn.Module):
    """
    队列嵌入层
    
    将队列大小、优先级、权重转换为状态向量
    """
    
    def __init__(self, queue_state_dim=32, max_num_queues=3):
        super().__init__()
        
        self.queue_state_dim = queue_state_dim
        self.max_num_queues = max_num_queues
        
        # 队列特征: 队列大小(1) + 优先级(one-hot, 3) + 权重(1) = 5
        num_queue_features = 1 + max_num_queues + 1
        
        # ========== TODO 13: 定义嵌入网络 ==========
        # 
        # self.net = nn.Sequential(...)
        self.net = nn.Sequential(
            nn.Linear(num_queue_features, queue_state_dim),
            nn.ReLU(),
            nn.Linear(queue_state_dim, queue_state_dim),
            nn.ReLU()
        )
        # ========================================
    
    def forward(self, queue_size, priority, weight):
        """
        前向传播
        
        参数:
            queue_size: 队列大小, shape [batch, num_queues]
            priority: 优先级索引, shape [batch, num_queues]
            weight: 权重, shape [batch, num_queues]
        
        返回:
            queue_state: 队列状态, shape [batch, num_queues, queue_state_dim]
        """
        batch_size, num_queues = queue_size.shape
        
        # ========== TODO 14: 对 priority 进行 one-hot 编码 ==========
        # 
        # priority_onehot = F.one_hot(...)
        priority_onehot = torch.nn.functional.one_hot(priority, num_classes=self.max_num_queues)
        # ============================================================
        
        # ========== TODO 15: 拼接所有队列特征 ==========
        # 
        # queue_size_unsqueezed = queue_size.unsqueeze(-1)
        # weight_unsqueezed = weight.unsqueeze(-1)
        # combined = torch.cat([...], dim=-1)
        queue_size_unsqueezed = queue_size.unsqueeze(-1)
        weight_unsqueezed = weight.unsqueeze(-1)
        combined = torch.cat([queue_size_unsqueezed, priority_onehot.float(), weight_unsqueezed], dim=-1)
        # =======================================================
        
        # ========== TODO 16: 通过嵌入网络 ==========
        # 
        # queue_state = self.net(combined)
        queue_state = self.net(combined)
        # =========================================
        
        return queue_state


def prepare_traffic_features(inputs):
    """
    准备流量特征并进行标准化
    """
    # Z-Score 标准化参数 (来自真实数据的统计)
    z_score = {
        'traffic': [1385.4, 859.8],
        'packets': [1.4, 0.89],
        'eq_lambda': [1350.9, 858.3],
        'avg_pkts_lambda': [0.91, 0.97],
        'exp_max_factor': [6.66, 4.71],
        'pkts_lambda_on': [0.91, 1.65],
        'avg_t_off': [1.66, 2.35],
        'avg_t_on': [1.66, 2.35],
        'ar_a': [0.0, 1.0],
        'sigma': [0.0, 1.0],
    }
    
    batch_size = inputs['traffic'].shape[0]
    num_paths = inputs['traffic'].shape[1]
    
    # 标准化每个特征
    traffic_list = []
    for key in ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda', 
                'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on',
                'ar_a', 'sigma']:
        x = inputs[key]
        mean = z_score[key][0]
        std = z_score[key][1]
        normalized = (x - mean) / std
        traffic_list.append(normalized)
    
    # 堆叠所有流量特征
    traffic_features = torch.stack(traffic_list, dim=-1)  # [batch, num_paths, 10]
    
    return traffic_features


def test_embeddings():
    """测试嵌入层"""
    print("\n" + "=" * 60)
    print("测试嵌入层")
    print("=" * 60)
    
    # 创建输入数据
    inputs = create_sample_inputs()
    
    print(f"\n输入数据:")
    for key, value in inputs.items():
        print(f"  {key}: {value.shape}")
    
    # 准备流量特征
    traffic_features = prepare_traffic_features(inputs)
    print(f"\n流量特征 (标准化后): {traffic_features.shape}")
    
    # 创建嵌入层
    path_embedding = PathEmbedding(path_state_dim=32, max_num_models=7)
    link_embedding = LinkEmbedding(link_state_dim=32, num_policies=4)
    queue_embedding = QueueEmbedding(queue_state_dim=32, max_num_queues=3)
    
    # 测试路径嵌入
    path_state = path_embedding(traffic_features, inputs['model'])
    print(f"\n路径状态: {path_state.shape}")
    
    # 测试链路嵌入
    link_state = link_embedding(inputs['capacity'], inputs['policy'])
    print(f"链路状态: {link_state.shape}")
    
    # 测试队列嵌入
    queue_state = queue_embedding(inputs['queue_size'], inputs['priority'], inputs['weight'])
    print(f"队列状态: {queue_state.shape}")
    
    # 验证形状
    print("\n验证形状:")
    print(f"  path_state [batch, num_paths, path_state_dim]: {list(path_state.shape)} ✓" if path_state.shape[2] == 32 else "  ✗")
    print(f"  link_state [batch, num_links, link_state_dim]: {list(link_state.shape)} ✓" if link_state.shape[2] == 32 else "  ✗")
    print(f"  queue_state [batch, num_queues, queue_state_dim]: {list(queue_state.shape)} ✓" if queue_state.shape[2] == 32 else "  ✗")


def main():
    print("=" * 60)
    print("Exercise 3.1: RouteNet-Fermi 嵌入层")
    print("=" * 60)
    
    test_embeddings()
    
    print("\n" + "=" * 60)
    print("练习完成！嵌入层将原始网络特征转换为向量表示")
    print("=" * 60)


if __name__ == "__main__":
    main()
