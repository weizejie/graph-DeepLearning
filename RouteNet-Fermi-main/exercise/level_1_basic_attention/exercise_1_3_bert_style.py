"""
Exercise 1.3: BERT 风格注意力层

目标：理解完整的 BERT 注意力层结构
- 多头注意力 + 残差连接 + 层归一化
- 前馈网络 (FFN) + 残差连接 + 层归一化

这与原始 BERT 论文中的 Transformer Encoder 结构一致

数据规模：
- seq_len = 5
- d_model = 16
- num_heads = 4
- d_ff = 32 (前馈网络隐藏层维度)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_sample_data():
    """创建简单的示例数据"""
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 5
    d_model = 16
    
    # 模拟 BERT 输入（已包含位置编码的词嵌入）
    x = torch.randn(batch_size, seq_len, d_model)
    
    return x


class LayerNorm(nn.Module):
    """层归一化 (Layer Normalization)"""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps
    
    def forward(self, x):
        # ========== TODO 1: 实现层归一化 ==========
        # 计算 x 的均值和标准差
        # 对最后一个维度 (d_model) 进行归一化
        # 最后应用 gamma 和 beta 进行缩放和偏移
        # 
        # mean = ...
        # std = ...
        # normalized = ...
        # output = ...
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        # =========================================
        return output


class BERTAttention(nn.Module):
    """BERT 风格的多头注意力 + 残差 + 层归一化"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 多头注意力的权重
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # 层归一化
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        
        # dropout (可选)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        """
        前向传播，包含残差连接和层归一化
        
        结构: 
        x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm
        """
        # ========== TODO 2: 实现自注意力 (带残差) ==========
        # 1. 计算 Q, K, V
        # 2. 分割多头
        # 3. 计算注意力
        # 4. 合并多头
        # 5. 输出投影
        # 6. dropout
        # 7. 残差连接: output = x + attention_output
        # 8. 层归一化
        # 
        # 提示: 参考之前的练习
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 分割多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影
        attention_output = self.W_O(context)
        attention_output = self.dropout(attention_output)
        
        # 残差连接 + 层归一化
        x = self.layernorm1(x + attention_output)
        # ==========================================================
        
        return x, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络 (Position-wise Feed-Forward Networks)"""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        # ========== TODO 3: 实现前馈网络 ==========
        # 两层线性变换 + ReLU 激活
        # 
        # self.linear1 = ...
        # self.activation = ...
        # self.linear2 = ...
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        # ========================================
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # ========== TODO 4: 前馈网络前向传播 ==========
        # x -> linear1 -> activation -> dropout -> linear2 -> dropout
        # 
        # ff_output = ...
        
        ff_output = self.linear1(x)
        ff_output = self.activation(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)
        ff_output = self.dropout(ff_output)
        # ============================================
        return ff_output


class BERTEncoderLayer(nn.Module):
    """完整的 BERT Encoder 层"""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        self.attention = BERTAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layernorm = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # ========== TODO 5: 实现完整的 Encoder 层流程 ==========
        # 1. 多头注意力 + 残差 + 层归一化
        # 2. 前馈网络 + 残差 + 层归一化
        # 
        # attention_output, attn_weights = self.attention(x, mask)
        # x = self.layernorm(attention_output)  # 已在 attention 内部完成
        # 
        # ff_output = self.feed_forward(x)
        # output = self.layernorm(x + ff_output)
        
        attention_output, attn_weights = self.attention(x, mask)
        ff_output = self.feed_forward(attention_output)
        output = self.layernorm(attention_output + ff_output)
        # ==========================================================
        return output, attn_weights


def main():
    print("=" * 60)
    print("Exercise 1.3: BERT 风格注意力层")
    print("=" * 60)
    
    # 创建数据
    x = create_sample_data()
    
    print(f"\n输入数据形状:")
    print(f"  x shape: {x.shape}")
    
    # 创建 BERT Encoder 层
    d_model = 16
    num_heads = 4
    d_ff = 32
    
    encoder = BERTEncoderLayer(d_model, num_heads, d_ff)
    
    # 运行
    output, attention_weights = encoder(x)
    
    print(f"\n输出数据形状:")
    print(f"  output shape: {output.shape}")
    print(f"  attention_weights shape: {attention_weights.shape}")
    
    # 验证残差连接：检查输出与输入的差异
    diff = (output - x).abs().mean().item()
    print(f"\n输出与输入的平均差异 (残差连接效果): {diff:.4f}")
    
    # 展示注意力权重
    print(f"\n注意力权重矩阵 (Head 0):")
    print(attention_weights[0, 0].detach().numpy())
    
    print("\n" + "=" * 60)
    print("练习完成！你已经完成了完整的 BERT Encoder 层！")
    print("=" * 60)


if __name__ == "__main__":
    main()
