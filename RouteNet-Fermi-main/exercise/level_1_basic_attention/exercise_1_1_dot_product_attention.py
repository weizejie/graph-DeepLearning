"""
Exercise 1.1: 点积注意力 (Dot Product Attention)

目标：理解最基础的注意力分数计算过程
- 计算 Q 和 K 的点积得到注意力分数
- 使用 softmax 将分数归一化
- 用归一化的分数对 V 进行加权求和

数据规模：
- seq_len = 3  (序列长度)
- d_k = 4      (每个头的维度)
- batch_size = 2
"""

import torch
import torch.nn.functional as F
import math


def create_sample_data():
    """创建简单的示例数据"""
    # 固定随机种子以便复现
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 3
    d_k = 4
    
    # 模拟 Q, K, V - 实际应用中这些来自输入的线性变换
    # 这里直接创建简单的数值便于理解
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    return Q, K, V


def dot_product_attention(Q, K, V):
    """
    基础点积注意力实现
    
    参数:
        Q: Query tensor, shape [batch, seq_len, d_k]
        K: Key tensor, shape [batch, seq_len, d_k]
        V: Value tensor, shape [batch, seq_len, d_k]
    
    返回:
        output: 加权后的输出, shape [batch, seq_len, d_k]
        attention_weights: 注意力权重, shape [batch, seq_len, seq_len]
    """
    batch_size, seq_len, d_k = Q.shape
    
    # ========== TODO 1: 计算注意力分数 ==========
    # 计算 Q 和 K 的点积
    # 提示: 使用 torch.matmul 或 @ 运算符
    # 注意: K 需要转置以进行矩阵乘法
    # 
    # scores = ...
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # =============================================
    
    # ========== TODO 2: 缩放点积 ==========
    # 为什么要除以 sqrt(d_k)？请在注释中解释
    # 
    # scaled_scores = ...
    scaled_scores = scores / math.sqrt(d_k)
    # =============================================
    
    # ========== TODO 3: 计算注意力权重 ==========
    # 使用 softmax 对最后一个维度进行归一化
    # 提示: F.softmax 或 torch.softmax
    # 
    # attention_weights = ...
    attention_weights = F.softmax(scaled_scores, dim=-1)
    # =============================================
    
    # ========== TODO 4: 加权求和 ==========
    # 用注意力权重对 V 进行加权求和
    # 
    # output = ...
    output = torch.matmul(attention_weights, V)
    # =============================================
    
    return output, attention_weights


def main():
    print("=" * 60)
    print("Exercise 1.1: 点积注意力 (Dot Product Attention)")
    print("=" * 60)
    
    # 创建数据
    Q, K, V = create_sample_data()
    
    print(f"\n输入数据形状:")
    print(f"  Q shape: {Q.shape}  (batch={Q.shape[0]}, seq_len={Q.shape[1]}, d_k={Q.shape[2]})")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    
    # 运行注意力机制
    output, attention_weights = dot_product_attention(Q, K, V)
    
    print(f"\n输出数据形状:")
    print(f"  output shape: {output.shape}")
    print(f"  attention_weights shape: {attention_weights.shape}")
    
    # 展示注意力权重（取第一个样本）
    print(f"\n注意力权重矩阵 (batch=0):")
    print(attention_weights[0].detach().numpy())
    
    # 验证：每行的和应该等于 1
    row_sums = attention_weights[0].sum(dim=-1)
    print(f"\n每行权重之和 (应该接近 1.0):")
    print(row_sums.detach().numpy())
    
    print("\n" + "=" * 60)
    print("练习完成！尝试修改 d_k 或 seq_len 来观察变化")
    print("=" * 60)


if __name__ == "__main__":
    main()
