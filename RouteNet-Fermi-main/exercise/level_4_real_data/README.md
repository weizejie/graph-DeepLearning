# Level 4: 真实数据处理

## 练习目标

学会使用 RouteNet-Fermi 原始仓库中的 DatanetAPI 处理真实数据集。

## 练习内容

本练习分为 2 个小节：

1. `exercise_4_1_datanet_api.py` - 理解 DatanetAPI 的数据结构
2. `exercise_4_2_training_loop.py` - 完整的训练循环

## 数据集说明

RouteNet-Fermi 使用 **Datanet** 数据集，包含：

| 数据集 | 描述 |
|--------|------|
| Fat-Tree | 胖树拓扑网络 |
| Testbed | 真实网络实验床 |
| Real Traffic | 真实流量数据 |

每个数据集包含：
- `.gz` 压缩文件：网络拓扑和流量矩阵
- 性能标签：延迟、抖动、丢包率

## DatanetAPI 数据结构

```python
{
    # 流量特征
    'traffic': tensor,      # 流量矩阵
    'packets': tensor,      # 数据包数量
    'length': tensor,       # 数据包长度
    'model': tensor,        # 路由模型类型
    
    # 流量统计特征
    'eq_lambda': ...,
    'avg_pkts_lambda': ...,
    'exp_max_factor': ...,
    'pkts_lambda_on': ...,
    'avg_t_off': ...,
    'avg_t_on': ...,
    'ar_a': ...,
    'sigma': ...,
    
    # 链路特征
    'capacity': tensor,     # 链路容量
    'policy': tensor,       # 调度策略
    'queue_size': tensor,  # 队列大小
    'priority': tensor,    # 优先级
    'weight': tensor,      # 权重
    
    # 拓扑关系
    'queue_to_path': tensor,
    'link_to_path': tensor,
    'path_to_link': tensor,
    'path_to_queue': tensor,
    'queue_to_link': tensor,
}
```

## 运行方式

```bash
cd exercise/level_4_real_data
python exercise_4_1_datanet_api.py
python exercise_4_2_training_loop.py
```

## 练习目标

1. **Exercise 4.1**: 理解 DatanetAPI 的数据加载和解析
2. **Exercise 4.2**: 完整的训练循环：数据加载 → 模型前向 → 损失计算 → 反向传播

## 前置要求

需要安装以下依赖：
```bash
pip install tensorflow-gpu==2.12.0  # 或 tensorflow
```

## 数据准备

数据集文件需要放置在对应的目录中：
- `fat_tree/train/` - 训练数据
- `fat_tree/test/` - 测试数据
