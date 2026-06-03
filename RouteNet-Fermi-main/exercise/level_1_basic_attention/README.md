# Level 1: 基础注意力机制练习

## 练习目标
理解 Self-Attention（自注意力机制）的工作原理，掌握 Q、K、V 的概念和计算过程。

## 练习内容

本练习分为 3 个小节，逐步深入：
1. `exercise_1_1_dot_product_attention.py` - 理解点积注意力
2. `exercise_1_2_multihead_attention.py` - 理解多头注意力
3. `exercise_1_3_bert_style.py` - 理解 BERT 中的注意力应用

## 数据说明

本级别使用简单的合成数据：
- 序列长度 (seq_len): 3-5
- 批量大小 (batch_size): 2
- 隐藏维度 (d_model): 8 或 16
- 头数 (num_heads): 2 或 4

## 运行方式

```bash
cd exercise/level_1_basic_attention
python exercise_1_1_dot_product_attention.py
python exercise_1_2_multihead_attention.py
python exercise_1_3_bert_style.py
```

## 练习顺序建议

按顺序完成，每个 exercise 会引入新的概念：

1. **Exercise 1.1**: 理解最基础的注意力分数计算（Q·K^T）
2. **Exercise 1.2**: 理解如何将多个注意力头组合
3. **Exercise 1.3**: 理解完整的 BERT 风格注意力层

## 提示

每个文件都有 `# TODO:` 标记，提示你需要填写的地方。

## PyTorch 内置方法参考

详细的 PyTorch 方法说明请参考：[PYTORCH_METHODS.md](./PYTORCH_METHODS.md)

该文件包含：
- 张量创建方法 (`torch.randn`, `torch.tensor`, 等)
- 数学运算方法 (`torch.matmul`, `torch.sum`, `torch.mean`, 等)
- 张量变换方法 (`.view`, `.transpose`, `.contiguous`, 等)
- 神经网络层 (`nn.Linear`, `nn.Sequential`, `nn.LayerNorm`, 等)
- 其他常用方法 (`F.softmax`, `F.one_hot`, `.detach`, 等)
- 常见错误及解决方案