"""
Exercise 4.1: DatanetAPI 数据加载

目标：理解如何使用 DatanetAPI 加载真实的网络数据集

DatanetAPI 是 RouteNet-Fermi 官方使用的数据加载工具，
可以解析 Datanet 数据集（.gz 压缩文件）。

本练习演示如何：
1. 加载 Datanet 数据集
2. 理解数据结构和字段
3. 提取拓扑关系和特征

注意：本练习主要展示数据加载流程，不需要真实的数据文件也能运行示例代码。
"""

import os
import sys
import numpy as np

# 为了演示，我们先创建一个模拟的 DatanetAPI
# 实际使用时，应该从 fat_tree 目录导入


class MockDatanetAPI:
    """
    模拟 DatanetAPI，用于演示数据结构
    
    真实情况下，应该使用：
    from datanetAPI import DatanetAPI
    """
    
    def __init__(self, data_dir, shuffle=False, seed=None):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.seed = seed
        
        # 模拟样本数量
        self.num_samples = 100
        
    def __iter__(self):
        """返回数据迭代器"""
        for i in range(self.num_samples):
            yield self._generate_sample(i)
    
    def _generate_sample(self, idx):
        """生成一个模拟样本"""
        np.random.seed(self.seed + idx if self.seed else idx)
        
        # 随机生成网络规模
        num_flows = np.random.randint(5, 20)  # 流/路径数量
        num_links = np.random.randint(3, 10)  # 链路数量
        num_queues = num_links * 2  # 每个链路2个队列（高/低优先级）
        
        return {
            'idx': idx,
            'num_flows': num_flows,
            'num_links': num_links,
            'num_queues': num_queues,
            # 流量特征
            'traffic': np.random.rand(num_flows) * 1000 + 100,
            'packets': np.random.rand(num_flows) * 10 + 1,
            'length': np.random.rand(num_flows) * 500 + 64,
            'model': np.random.randint(0, 7, num_flows),
            # 流量统计
            'eq_lambda': np.random.rand(num_flows) * 100 + 10,
            'avg_pkts_lambda': np.random.rand(num_flows) * 2,
            'exp_max_factor': np.random.rand(num_flows) * 5 + 1,
            'pkts_lambda_on': np.random.rand(num_flows) * 2,
            'avg_t_off': np.random.rand(num_flows) * 2 + 0.5,
            'avg_t_on': np.random.rand(num_flows) * 2 + 0.5,
            'ar_a': np.random.rand(num_flows),
            'sigma': np.random.rand(num_flows) * 0.5,
            # 链路特征
            'capacity': np.random.rand(num_links) * 10000 + 1000,
            'policy': np.random.randint(0, 4, num_links),
            # 队列特征
            'queue_size': np.random.rand(num_queues) * 10000 + 1000,
            'priority': np.random.randint(0, 3, num_queues),
            'weight': np.random.rand(num_queues) * 3 + 1,
            # 拓扑关系（简化的随机邻接）
            'path_to_link': [np.random.choice(num_links, np.random.randint(1, 4), replace=False).tolist() 
                            for _ in range(num_flows)],
            'link_to_path': [np.random.choice(num_flows, np.random.randint(1, 5), replace=False).tolist()
                            for _ in range(num_links)],
            'path_to_queue': [np.random.choice(num_queues, np.random.randint(1, 3), replace=False).tolist()
                             for _ in range(num_flows)],
            'queue_to_path': [np.random.choice(num_flows, np.random.randint(1, 3), replace=False).tolist()
                             for _ in range(num_queues)],
            'queue_to_link': [i % num_links for i in range(num_queues)],
            # 性能标签
            'delay': np.random.rand(num_flows) * 100 + 10,  # 延迟 (ms)
            'jitter': np.random.rand(num_flows) * 10,  # 抖动
            'loss': np.random.rand(num_flows) * 0.01,  # 丢包率
        }


def explore_data_structure():
    """探索数据结构"""
    print("=" * 60)
    print("DatanetAPI 数据结构探索")
    print("=" * 60)
    
    # 模拟加载数据
    data_api = MockDatanetAPI(data_dir="./data", shuffle=False, seed=42)
    
    # 获取第一个样本
    sample = next(iter(data_api))
    
    print(f"\n样本索引: {sample['idx']}")
    print(f"网络规模:")
    print(f"  路径/流数量: {sample['num_flows']}")
    print(f"  链路数量: {sample['num_links']}")
    print(f"  队列数量: {sample['num_queues']}")
    
    print(f"\n流量特征 (Traffic Features):")
    print(f"  traffic (流量速率): shape = {sample['traffic'].shape}")
    print(f"  packets (数据包数量): shape = {sample['packets'].shape}")
    print(f"  length (数据包长度): shape = {sample['length'].shape}")
    print(f"  model (路由模型): shape = {sample['model'].shape}")
    
    print(f"\n流量统计特征 (Traffic Statistics):")
    stats_features = ['eq_lambda', 'avg_pkts_lambda', 'exp_max_factor', 
                      'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma']
    for key in stats_features:
        print(f"  {key}: shape = {sample[key].shape}")
    
    print(f"\n链路特征 (Link Features):")
    print(f"  capacity (链路容量): shape = {sample['capacity'].shape}")
    print(f"  policy (调度策略): shape = {sample['policy'].shape}")
    
    print(f"\n队列特征 (Queue Features):")
    print(f"  queue_size (队列大小): shape = {sample['queue_size'].shape}")
    print(f"  priority (优先级): shape = {sample['priority'].shape}")
    print(f"  weight (权重): shape = {sample['weight'].shape}")
    
    print(f"\n拓扑关系 (Topology):")
    print(f"  path_to_link: {sample['path_to_link'][:2]}...")
    print(f"  link_to_path: {sample['link_to_path'][:2]}...")
    print(f"  path_to_queue: {sample['path_to_queue'][:2]}...")
    print(f"  queue_to_path: {sample['queue_to_path'][:2]}...")
    print(f"  queue_to_link: {sample['queue_to_link'][:2]}...")
    
    print(f"\n性能标签 (Performance):")
    print(f"  delay (延迟): shape = {sample['delay'].shape}")
    print(f"  jitter (抖动): shape = {sample['jitter'].shape}")
    print(f"  loss (丢包率): shape = {sample['loss'].shape}")


def demonstrate_ragged_tensors():
    """
    演示 Ragged Tensor（不规则张量）的概念
    
    在 RouteNet-Fermi 中：
    - 每个样本的路径数量不同
    - 每个路径经过的链路/队列数量不同
    - 需要使用 ragged tensor 处理
    """
    print("\n" + "=" * 60)
    print("Ragged Tensor 演示")
    print("=" * 60)
    
    # 模拟两个样本
    sample1 = {
        'num_flows': 5,
        'path_to_link': [[0, 1], [1, 2], [0, 2], [1], [0, 1, 2]],
    }
    
    sample2 = {
        'num_flows': 3,
        'path_to_link': [[0], [0, 1], [1]],
    }
    
    print(f"\n样本1: {sample1['num_flows']} 条路径")
    print(f"  路径0 经过链路: {sample1['path_to_link'][0]}")
    print(f"  路径1 经过链路: {sample1['path_to_link'][1]}")
    print(f"  ...")
    
    print(f"\n样本2: {sample2['num_flows']} 条路径")
    print(f"  路径0 经过链路: {sample2['path_to_link'][0]}")
    print(f"  路径1 经过链路: {sample2['path_to_link'][1]}")
    
    print("""
问题：每个路径经过的链路数量不同，如何处理？

解决方案：
1. 使用填充（Padding）：将所有路径填充到相同长度
2. 使用 Mask：标记哪些是有效数据
3. 使用 Ragged Tensor：TF/Keras 原生支持

在 RouteNet-Fermi 中使用 tf.RaggedTensor 来处理！
""")


def real_api_example():
    """
    真实 DatanetAPI 使用示例
    
    实际使用时，应该这样加载数据：
    """
    print("\n" + "=" * 60)
    print("真实 DatanetAPI 使用示例")
    print("=" * 60)
    
    example_code = '''
# ========== TODO 1: 导入 DatanetAPI ==========
# 
# from datanetAPI import DatanetAPI
# 
# =============================================

# ========== TODO 2: 初始化数据加载器 ==========
# 
# tool = DatanetAPI(
#     data_dir='../fat_tree/train',  # 数据目录
#     shuffle=True,                    # 是否打乱
#     seed=42                          # 随机种子
# )
# 
# =============================================

# ========== TODO 3: 遍历数据 ==========
# 
# for sample in tool:
#     # 获取拓扑
#     G = nx.DiGraph(sample.get_topology_object())
#     
#     # 获取流量矩阵
#     T = sample.get_traffic_matrix()
#     
#     # 获取路由矩阵
#     R = sample.get_routing_matrix()
#     
#     # 获取性能矩阵
#     P = sample.get_performance_matrix()
#     
#     # 处理数据...
# 
# =============================================
'''
    print(example_code)


def convert_to_tensors(sample):
    """
    将样本转换为 TensorFlow/PyTorch 张量
    
    这是数据预处理的关键步骤！
    """
    print("\n" + "=" * 60)
    print("数据转换为张量")
    print("=" * 60)
    
    # 模拟转换
    print("""
转换步骤：
1. 流量特征 → 标准化 (Z-Score)
2. 分类特征 → One-Hot 编码
3. 拓扑关系 → 索引张量
4. 标签 → 保持原始值或标准化

示例代码 (PyTorch):
```python
import torch

# 流量特征标准化
traffic_mean = 1385.4
traffic_std = 859.8
traffic_normalized = (traffic - traffic_mean) / traffic_std

# 路由模型 one-hot
model_onehot = F.one_hot(model, num_classes=7)

# 拓扑关系转换为张量
path_to_link = torch.tensor(path_to_link)  # 不规则，需要处理
```

示例代码 (TensorFlow):
```python
import tensorflow as tf

# 流量特征标准化
traffic_normalized = (traffic - 1385.4) / 859.8

# 拓扑关系 - 使用 Ragged Tensor
path_to_link = tf.ragged.constant(path_to_link)
```
""")


def main():
    print("=" * 60)
    print("Exercise 4.1: DatanetAPI 数据加载")
    print("=" * 60)
    
    # 1. 探索数据结构
    explore_data_structure()
    
    # 2. 理解 ragged tensor
    demonstrate_ragged_tensors()
    
    # 3. 真实 API 示例
    real_api_example()
    
    # 4. 数据转换
    convert_to_tensors(None)
    
    print("\n" + "=" * 60)
    print("练习完成！理解数据加载是使用 RouteNet-Fermi 的第一步")
    print("=" * 60)
    print("""
提示：
- 真实数据集需要从 Datanet 官网下载
- 或者使用项目中已有的 fat_tree 数据集
- 理解数据结构后，可以开始训练模型
""")


if __name__ == "__main__":
    main()
