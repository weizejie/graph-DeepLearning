"""
Exercise 4.2: 完整的训练循环

目标：理解 RouteNet-Fermi 的完整训练流程

本练习演示：
1. 数据准备和批处理
2. 模型前向传播
3. 损失计算（MSELoss）
4. 反向传播和优化
5. 评估指标（MAE, RMSE）

这是一个完整的 PyTorch 训练循环示例！
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# ==================== 数据准备 ====================

class SimpleNetworkDataset(Dataset):
    """
    简单的网络数据集
    
    实际使用时，应该从 DatanetAPI 加载真实数据
    """
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = self._generate_data()
    
    def _generate_data(self):
        """生成模拟数据"""
        torch.manual_seed(42)
        data = []
        
        for i in range(self.num_samples):
            # 随机网络规模
            num_paths = np.random.randint(3, 10)
            num_links = np.random.randint(2, 6)
            num_queues = num_links * 2
            
            # 流量特征 (10维)
            traffic_features = torch.randn(num_paths, 10) * 0.5
            
            # 模型 (分类特征)
            model = torch.randint(0, 7, (num_paths,))
            
            # 链路特征
            capacity = torch.randn(num_links) * 0.5 + 0.5  # 标准化后
            policy = torch.randint(0, 4, (num_links,))
            
            # 队列特征
            queue_size = torch.randn(num_queues) * 0.5 + 0.5
            priority = torch.randint(0, 3, (num_queues,))
            weight = torch.rand(num_queues)
            
            # 拓扑关系
            path_to_link = [list(np.random.choice(num_links, np.random.randint(1, 4), replace=False)) 
                          for _ in range(num_paths)]
            link_to_path = [list(np.random.choice(num_paths, np.random.randint(1, 4), replace=False)) 
                           for _ in range(num_links)]
            path_to_queue = [list(np.random.choice(num_queues, np.random.randint(1, 3), replace=False)) 
                            for _ in range(num_paths)]
            queue_to_path = [list(np.random.choice(num_paths, np.random.randint(1, 3), replace=False)) 
                            for _ in range(num_queues)]
            queue_to_link = [i % num_links for i in range(num_queues)]
            
            # 标签 (延迟)
            delay = torch.randn(num_paths).abs() * 10 + 5  # 5-15ms
            
            # 路径数量 (用于后续处理)
            length = torch.tensor([num_paths])
            
            data.append({
                'traffic': traffic_features,
                'model': model,
                'capacity': capacity,
                'policy': policy,
                'queue_size': queue_size,
                'priority': priority,
                'weight': weight,
                'path_to_link': path_to_link,
                'link_to_path': link_to_path,
                'path_to_queue': path_to_queue,
                'queue_to_path': queue_to_path,
                'queue_to_link': queue_to_link,
                'delay': delay,
                'length': length,
            })
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==================== 模型定义 ====================

class PathEmbedding(nn.Module):
    """路径嵌入层"""
    def __init__(self, path_state_dim=32, max_num_models=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10 + max_num_models, path_state_dim),
            nn.ReLU(),
            nn.Linear(path_state_dim, path_state_dim),
            nn.ReLU()
        )
    
    def forward(self, traffic_features, model):
        model_onehot = torch.nn.functional.one_hot(model, num_classes=7).float()
        combined = torch.cat([traffic_features, model_onehot], dim=-1)
        return self.net(combined)


class LinkEmbedding(nn.Module):
    """链路嵌入层"""
    def __init__(self, link_state_dim=32):
        self.net = nn.Sequential(
            nn.Linear(1 + 4, link_state_dim),
            nn.ReLU(),
            nn.Linear(link_state_dim, link_state_dim),
            nn.ReLU()
        )
        super().__init__()
    
    def forward(self, capacity, policy):
        policy_onehot = torch.nn.functional.one_hot(policy, num_classes=4).float()
        combined = torch.cat([capacity.unsqueeze(-1), policy_onehot], dim=-1)
        return self.net(combined)


class QueueEmbedding(nn.Module):
    """队列嵌入层"""
    def __init__(self, queue_state_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 3 + 1, queue_state_dim),
            nn.ReLU(),
            nn.Linear(queue_state_dim, queue_state_dim),
            nn.ReLU()
        )
    
    def forward(self, queue_size, priority, weight):
        priority_onehot = torch.nn.functional.one_hot(priority, num_classes=3).float()
        combined = torch.cat([queue_size.unsqueeze(-1), priority_onehot, weight.unsqueeze(-1)], dim=-1)
        return self.net(combined)


class RouteNetFermi(nn.Module):
    """简化版 RouteNet-Fermi"""
    
    def __init__(self, path_state_dim=32, link_state_dim=32, queue_state_dim=32, iterations=4):
        super().__init__()
        
        self.path_embedding = PathEmbedding(path_state_dim)
        self.link_embedding = LinkEmbedding(link_state_dim)
        self.queue_embedding = QueueEmbedding(queue_state_dim)
        
        self.path_gru = nn.GRUCell(link_state_dim + queue_state_dim, path_state_dim)
        self.link_gru = nn.GRUCell(queue_state_dim, link_state_dim)
        self.queue_gru = nn.GRUCell(path_state_dim, queue_state_dim)
        
        self.readout = nn.Sequential(
            nn.Linear(path_state_dim, path_state_dim // 2),
            nn.ReLU(),
            nn.Linear(path_state_dim // 2, 1)
        )
        
        self.path_state_dim = path_state_dim
        self.link_state_dim = link_state_dim
        self.queue_state_dim = queue_state_dim
    
    def aggregate(self, state, neighbors, max_len):
        """聚合邻居状态"""
        batch_size = 1
        num_nodes = len(neighbors)
        aggregated = torch.zeros(batch_size, num_nodes, state.shape[-1])
        
        for i in range(num_nodes):
            neighbor_ids = neighbors[i]
            if len(neighbor_ids) > 0:
                neighbor_states = state[0, neighbor_ids, :]
                aggregated[0, i] = neighbor_states.sum(dim=0)
        
        return aggregated
    
    def forward(self, inputs):
        # 嵌入
        path_state = self.path_embedding(inputs['traffic'].unsqueeze(0), inputs['model'].unsqueeze(0))
        link_state = self.link_embedding(inputs['capacity'].unsqueeze(0), inputs['policy'].unsqueeze(0))
        queue_state = self.queue_embedding(inputs['queue_size'].unsqueeze(0), 
                                           inputs['priority'].unsqueeze(0), 
                                           inputs['weight'].unsqueeze(0))
        
        # 消息传递
        for _ in range(4):
            link_agg = self.aggregate(link_state, inputs['link_to_link'], 5)
            queue_agg = self.aggregate(queue_state, inputs['queue_to_path'], 3)
            
            path_input = torch.cat([link_agg, queue_agg], dim=-1)
            path_state_flat = path_state.view(1, -1)
            path_input_flat = path_input.view(1, -1)
            new_path_state = self.path_gru(path_input_flat, path_state_flat)
            path_state = new_path_state.view(1, -1, self.path_state_dim)
            
            path_to_queue_agg = self.aggregate(path_state, inputs['path_to_queue'], 3)
            queue_state_flat = queue_state.view(1, -1)
            new_queue_state = self.queue_gru(path_to_queue_agg.view(1, -1), queue_state_flat)
            queue_state = new_queue_state.view(1, -1, self.queue_state_dim)
            
            queue_to_link_agg = self.aggregate(queue_state, inputs['queue_to_link'], 5)
            link_state_flat = link_state.view(1, -1)
            new_link_state = self.link_gru(queue_to_link_agg.view(1, -1), link_state_flat)
            link_state = new_link_state.view(1, -1, self.link_state_dim)
        
        # Readout
        delay_pred = self.readout(path_state)
        delay_pred = delay_pred.squeeze(-1).sum(dim=-1)
        
        return delay_pred


# ==================== 训练函数 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in dataloader:
        # ========== TODO 1: 将数据移动到设备 ==========
        # 
        # 将每个 tensor 移动到 GPU/CPU
        # 
        batch_data = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(device)
            else:
                batch_data[key] = value
        # ============================================
        
        # ========== TODO 2: 前向传播 ==========
        # 
        # output = model(batch_data)
        # 
        output = model(batch_data)
        # ============================================
        
        # ========== TODO 3: 计算损失 ==========
        # 
        # target = batch_data['delay']
        # loss = criterion(output, target)
        target = batch_data['delay'].sum()  # 简化：预测总延迟
        loss = criterion(output, target.unsqueeze(0))
        # ============================================
        
        # ========== TODO 4: 反向传播 ==========
        # 
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ============================================
        
        total_loss += loss.item()
        num_samples += 1
    
    return total_loss / num_samples


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            batch_data = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device)
                else:
                    batch_data[key] = value
            
            # 前向传播
            output = model(batch_data)
            target = batch_data['delay'].sum()
            
            # 计算损失
            loss = criterion(output, target.unsqueeze(0))
            total_loss += loss.item()
            
            # 计算 MAE
            mae = torch.abs(output - target).item()
            total_mae += mae
            
            num_samples += 1
    
    return {
        'loss': total_loss / num_samples,
        'mae': total_mae / num_samples,
        'rmse': math.sqrt(total_loss / num_samples)
    }


def collate_fn(batch):
    """自定义批处理函数"""
    return batch[0]  # 简化：每次只处理一个样本


def main():
    print("=" * 60)
    print("Exercise 4.2: 完整的训练循环")
    print("=" * 60)
    
    # 超参数
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 5
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # ========== TODO 5: 创建数据集和数据加载器 ==========
    # 
    # train_dataset = SimpleNetworkDataset(...)
    # train_loader = DataLoader(...)
    # 
    train_dataset = SimpleNetworkDataset(num_samples=100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # ====================================================
    
    # ========== TODO 6: 创建模型 ==========
    # 
    # model = RouteNetFermi(...).to(device)
    # 
    model = RouteNetFermi(
        path_state_dim=32,
        link_state_dim=32,
        queue_state_dim=32,
        iterations=4
    ).to(device)
    # ========================================
    
    # ========== TODO 7: 定义损失函数和优化器 ==========
    # 
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(...)
    # 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # =========================================================
    
    print(f"\n数据集大小: {len(train_dataset)}")
    print(f"批量大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 评估
        metrics = evaluate(model, train_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {metrics['loss']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    print("""
训练流程总结：
1. 数据加载：使用 Dataset 和 DataLoader
2. 模型前向：输入数据 → 嵌入 → 消息传递 → Readout
3. 损失计算：MSE Loss (预测值 vs 真实值)
4. 反向传播：loss.backward() 计算梯度
5. 参数更新：optimizer.step() 更新权重

下一步：
- 使用真实 DatanetAPI 加载数据
- 实现完整的 RouteNet-Fermi 模型
- 调整超参数优化性能
""")


if __name__ == "__main__":
    main()
