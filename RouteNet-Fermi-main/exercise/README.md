# RouteNet-Fermi 练习目录

本目录包含一系列由浅入深的练习，帮助你理解 RouteNet-Fermi 的核心概念和实现原理。

## 练习结构

```
exercise/
├── level_1_basic_attention/     # 基础注意力机制
│   ├── exercise_1_1_dot_product_attention.py
│   ├── exercise_1_2_multihead_attention.py
│   └── exercise_1_3_bert_style.py
│
├── level_2_message_passing/     # 图消息传递
│   ├── exercise_2_1_simple_graph.py
│   ├── exercise_2_2_message_passing.py
│   └── exercise_2_3_gru_message_passing.py
│
├── level_3_routenet_basic/      # RouteNet-Fermi 基础
│   ├── exercise_3_1_embedding.py
│   └── exercise_3_2_full_model.py
│
└── level_4_real_data/           # 真实数据处理
    ├── exercise_4_1_datanet_api.py
    └── exercise_4_2_training_loop.py
```

## 练习顺序

### Level 1: 基础注意力机制
理解 Transformer/BERT 中的核心机制：
- 点积注意力
- 多头注意力
- 残差连接和层归一化

### Level 2: 图消息传递
理解 RouteNet-Fermi 的核心思想：
- 图结构的表示方法
- 消息传递机制
- GRU 状态更新

### Level 3: RouteNet-Fermi 基础
完整理解 RouteNet-Fermi：
- 特征嵌入层
- 完整的前向传播
- Readout 层

### Level 4: 真实数据处理
实践完整流程：
- DatanetAPI 数据加载
- 训练循环实现

## 如何使用

1. **按顺序进行**：建议从 Level 1 开始，逐步深入

2. **运行练习**：
   ```bash
   cd exercise/level_1_basic_attention
   python exercise_1_1_dot_product_attention.py
   ```

3. **填写 TODO**：
   - 每个文件都有 `# TODO:` 标记
   - 这些是需要你填写代码的地方
   - 先尝试运行，然后根据提示填代码

## 练习特点

- **渐进式难度**：从简单的注意力机制到完整的模型
- **挖坑式学习**：故意留有缺口，让你动手实践
- **代码可运行**：每个练习都可以独立运行
- **数据简化**：从合成数据开始，逐步过渡到真实数据

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy
- (可选) TensorFlow 2.x (用于真实数据处理)

安装依赖：
```bash
pip install torch numpy
```

## 参考资料

- [RouteNet-Fermi 论文](https://github.com/knowledgedefinednetworking/RouteNet-Fermi)
- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## 练习目标

完成所有练习后，你将：
1. ✅ 理解注意力机制的工作原理
2. ✅ 理解图神经网络消息传递
3. ✅ 掌握 RouteNet-Fermi 的完整架构
4. ✅ 能够使用真实数据训练模型
