"""
Exercise 1.2: 多头注意力 (Multi-Head Attention)

目标：理解多头注意力的工作原理
- 将输入分割成多个头
- 每个头独立计算注意力
- 合并多个头的输出

数据规模：
- seq_len = 3
- d_model = 8
- num_heads = 2
- d_k = d_model / num_heads = 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_sample_data():
    """创建简单的示例数据"""
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 3
    d_model = 8
    
    # 模拟输入 (在真实 BERT 中，这来自 Embedding 层)
    x = torch.randn(batch_size, seq_len, d_model)
    
    return x


class MultiHeadAttention(nn.Module):
    """多头注意力模块（简化版）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # ========== TODO 1: 定义 Q, K, V 的线性投影矩阵 ==========
        # 使用 nn.Linear 创建可学习的权重矩阵
        # 输入和输出维度都是 d_model
        # 
        # self.W_Q = ...
        # self.W_K = ...
        # self.W_V = ...
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # ==========================================================
        
        # ========== TODO 2: 定义输出投影矩阵 ==========
        # 将多头的结果合并后，再进行一次线性变换
        # 
        # self.W_O = ...
        self.W_O = nn.Linear(d_model, d_model)
        # =============================================
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入 tensor, shape [batch, seq_len, d_model]
            mask: 可选的 mask tensor
        
        返回:
            output: 多头注意力的输出, shape [batch, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # ========== TODO 3: 线性投影 ==========
        # 将输入 x 通过三个权重矩阵得到 Q, K, V
        # 
        # Q = ...
        # K = ...
        # V = ...
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # =============================================
        
        # ========== TODO 4: 分割成多个头 ==========
        # 将 Q, K, V 从 [batch, seq_len, d_model] 
        # 变换为 [batch, num_heads, seq_len, d_k]
        # 
        # 提示: 使用 view() 和 transpose()
        # 
        # Q = Q.view(...).transpose(...)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # =============================================
        
        # ========== TODO 5: 计算注意力分数 ==========
        # 使用点积计算注意力分数
        # 
        # scores = torch.matmul(Q, K.transpose(...)) / ...
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # =============================================
        
        # ========== TODO 6: 应用 Mask (可选) ==========
        # 如果提供了 mask，将 mask 为 0 的位置设为很小的负数
        # 
        # if mask is not None:
        #     scores = ...
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # =============================================
        
        # ========== TODO 7: Softmax 归一化 ==========
        # 
        # attention_weights = ...
        attention_weights = F.softmax(scores, dim=-1)
        # =============================================
        
        # ========== TODO 8: 加权求和 ==========
        # 用注意力权重对 V 进行加权求和
        # 
        # context = torch.matmul(...)
        context = torch.matmul(attention_weights, V)
        # =============================================
        
        # ========== TODO 9: 合并多个头 ==========
        # 将 [batch, num_heads, seq_len, d_k] 变回 [batch, seq_len, d_model]
        # 
        # 提示: transpose -> view (需要 contiguous())
        # 
        # context = context.transpose(...).contiguous().view(...)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # =============================================
        
        # ========== TODO 10: 输出投影 ==========
        # 
        # output = self.W_O(context)
        output = self.W_O(context)
        # =============================================
        
        return output, attention_weights


def main():
    print("=" * 60)
    print("Exercise 1.2: 多头注意力 (Multi-Head Attention)")
    print("=" * 60)
    
    # 创建数据
    x = create_sample_data()
    
    print(f"\n输入数据形状:")
    print(f"  x shape: {x.shape}")
    
    # 创建多头注意力模块
    d_model = 8
    num_heads = 2
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 运行多头注意力
    output, attention_weights = mha(x)
    
    print(f"\n输出数据形状:")
    print(f"  output shape: {output.shape}")
    print(f"  attention_weights shape: {attention_weights.shape}")
    
    # 展示每个头的注意力权重
    print(f"\n注意力权重矩阵:")
    for head_idx in range(num_heads):
        print(f"\n  Head {head_idx}:")
        print(attention_weights[0, head_idx].detach().numpy())
    
    print("\n" + "=" * 60)
    print("练习完成！尝试添加不同的 mask 观察效果")
    print("=" * 60)


if __name__ == "__main__":
    main()
