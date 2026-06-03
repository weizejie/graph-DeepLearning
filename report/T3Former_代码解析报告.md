# T3Former 代码结构解析与 PyTorch 核心知识指南

> 本报告结合 T3Former 完整源码，系统解析其架构设计，同时穿插讲解 PyTorch 核心概念，帮你通过实战掌握 PyTorch 框架。

---

## 目录

1. [T3Former 整体架构概览](#1-t3former-整体架构概览)
2. [核心模块逐个解析](#2-核心模块逐个解析)
   - 2.1 `TransformerClassifier` — 标准 Transformer 编码器
   - 2.2 `SAGE` — 图神经网络分支
   - 2.3 `AttentionFusion` — 多模态注意力融合
   - 2.4 `T3Former` — 三分支融合主模型
3. [PyTorch 核心概念详解（代码对照）](#3-pytorch-核心概念详解代码对照)
   - 3.1 `torch.nn.Module` — 一切模型的基类
   - 3.2 `super().__init__()` 与模块注册
   - 3.3 `forward()` — 数据流向
   - 3.4 `nn.Parameter` — 可学习参数
   - 3.5 `torch.nn.ModuleList` / `ModuleDict`
   - 3.6 常用层：`Linear`、`Dropout`、`ReLU`、`BatchNorm`
   - 3.7 `nn.Transformer` — PyTorch 内置 Transformer
   - 3.8 `nn.MultiheadAttention`
   - 3.9 `nn.Sequential`
4. [训练流程解析](#4-训练流程解析)
5. [数据处理管线](#5-数据处理管线)
6. [架构总结图](#6-架构总结图)

---

## 1. T3Former 整体架构概览

T3Former 是一个**三分支多模态融合**的时序图分类模型，核心思想是用三种不同的特征提取器从不同角度捕获时序图数据的特征，然后通过**注意力融合**来加权组合这些特征。

### 三大分支

| 分支名称 | 类型 | 输入数据 | 作用 |
|---|---|---|---|
| `transformer_branch` | 标准 Transformer | Betti 数特征（拓扑不变量） | 捕获时序拓扑变化模式 |
| `transformer_branch2` | 标准 Transformer | DoS 特征（密度状态） | 捕获时序密度状态变化 |
| `sage_branch` | GraphSAGE | 图结构数据（节点/边） | 捕获空间图结构特征 |

### 完整数据流

```
输入数据
  ├── Betti 特征  ──→  TransformerClassifier  ──┐
  ├── DoS 特征    ──→  TransformerClassifier  ──┼──→ AttentionFusion ──→ FC → 分类结果
  └── 图数据     ──→  SAGE / GraphTransformer ──┘
```

---

## 2. 核心模块逐个解析

### 2.0 `model.py` 全模块一览

| 类名 | 文件 | 类型 | 作用 |
|---|---|---|---|
| `TransformerClassifier` | model.py | 分支模块 | 标准 Transformer 编码器，处理时序特征 |
| `CNNTransformer` | model.py | 分支模块 | CNN + Transformer，处理图像/网格数据 |
| `SAGE` | model.py | 分支模块 | GraphSAGE 图神经网络 |
| `AttentionFusion` | model.py | 融合模块 | 自注意力多模态融合 |
| `T3Former` | model.py | 主模型 | 三分支融合（核心模型） |
| `T3SAGE` | model.py | 主模型 | T3Former 简化版（拼接替代注意力融合） |
| `T3GT` | model.py | 主模型 | T3Former 变体（用 GraphTransformer 替代 SAGE） |
| `T3GNN` | model.py | 主模型 | T3Former 变体（使用交叉注意力） |
| `CrossAttention` | model.py | 注意力模块 | 交叉注意力（T3GNN 用） |
| `GNN` | model.py | 独立模型 | 通用 GNN（支持 GCN/SAGE/GAT/GIN/TAG/Cheb/ARMA） |
| `GraphTransformer` | model.py | 独立模型 | 基于 TransformerConv 的图神经网络 |
| `GPSModel` | model.py | 独立模型 | GPS 混合模型（GNN + 注意力） |
| `Graphormer` | model.py | 独立模型 | Graphormer 变体（含度编码） |
| `NeuroGraphDynamic` | NeuroGraph.py | 数据集类 | 神经影像动态图数据集加载器 |
| Betti/DoS 特征提取 | dos_betti_features_*.py | 特征工程 | 计算拓扑不变量和密度状态特征 |
| `train_GNN.py` | train_GNN.py | 训练脚本 | GNN 模型对比实验（支持 8 种模型） |

---

### 2.1 `TransformerClassifier` — 标准 Transformer 编码器

```7:23:model.py
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps, drop_out):
        # 1. 继承nn.Module, 保证自定义的模型可以使用PyTorch的参数管理、转device等功能
        super(TransformerClassifier, self).__init__()

        # 2. 定义时序特征输入嵌入层
        # nn.Linear是最基础的全连接层，这里将输入的各时间步特征（input_dim）投影到hidden_dim维
        # 常见用法：nn.Linear(in_features, out_features)
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 3. 定义可学习的位置编码（Positional Encoding）
        # 形状为(1, num_timesteps, hidden_dim)：一个batch内所有样本都共享这组位置编码
        # 这里的 nn.Parameter 代码详解：
        # nn.Parameter 的作用是将普通的张量（tensor）注册为模型的“可学习参数”。
        # 只要是 nn.Parameter，pytorch 会自动把这部分参数收进 model.parameters()（即优化器会更新它）。
        # 例子：下面一行新建了一个全 0 的张量，尺寸是 [1, num_timesteps, hidden_dim]，代表所有 batch 共享的一组位置编码（即每个时间步都能有一组独立 learnable 的特征向量）。
        # 通过 nn.Parameter 包装后，这个 pos_encoding 能随着训练实时更新、参与反向传播。
        # 原理总结：（1）让张量参与梯度计算 --> (2) 自动注册进模型参数 --> (3) 优化器.step() 时同步更新。
        # 代码如下：
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        # 这样模型在训练中能自动学习“时间顺序”相关的最佳编码。例如在时序任务里，相邻步骤的影响或周期性信息，可全部融入进这组可学习的位置向量。
        # nn.Parameter 参考文档：https://pytorch.org/docs/stable/generated/torch.nn.Parameter.html
   

        # 4. 定义Transformer编码器
        # nn.Transformer是PyTorch原生的Transformer编码-解码器实现，不依赖第三方库
        self.transformer = nn.Transformer(
            d_model=hidden_dim,          # 每个token的特征维度。这里d_model=hidden_dim即嵌入后的特征长度
            nhead=n_heads,               # Multi-head Attention的头数（让注意力机制关注多处子空间）
            num_encoder_layers=n_layers, # 编码器堆叠层数
            num_decoder_layers=n_layers  # 解码器层数（本案例只用encoder，decoder不会实际调用）
        )
        # ➡️ 官方文档：https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # nn.Transformer具有encoder和decoder，但时序特征分类只用encoder做特征聚合

        # 5. 定义输出全连接层
        # 将最后输出的所有时间步的特征拼接(flatten)后，送入一个全连接层做分类/回归
        # 输入维度：hidden_dim*num_timesteps，输出维度：output_dim（类别数或目标维数）
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)

        # 6. 定义Dropout，提升模型泛化能力，防止过拟合
        # nn.Dropout(p)：对输入activations以概率p置零
        self.dropout = nn.Dropout(p=drop_out)
   

    def forward(self, src):
        # src: 输入的时序特征张量，形状为 (batch_size, num_timesteps, input_dim)
        # 逐步讲解：

        # 第一步：通过线性层 self.embedding 将输入特征映射到 hidden_dim 维空间
        # 并加上可学习的位置编码 self.positional_encoding（形状[1, 时间步数, hidden_dim]）
        # 这样让模型能够感知“每个时间点”的顺序信息
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]

        ###### 教学讲解 ######

        # ① 这行代码实现了什么？
        #   - self.embedding(src)：用一个全连接层（线性层）把原始的每个时间点的输入特征（input_dim维度）
        #     投影/变换到隐藏空间（hidden_dim维度），让模型有更强的表达力；
        #   - self.positional_encoding[:, :src.size(1), :]：取出shape为[1, 时间步, hidden_dim]的可学习位置编码张量，
        #     让模型知道每个时间点的“先后顺序”。
        #   - 两项相加：让embedding后每个时刻的特征都带上了该时刻的“位置信息”。

        # ② 为什么要这样写？（格式的意义是什么？）
        #   - Transformer结构原本用于自然语言，每个token顺序很重要。
        #   - 时序数据也有“谁在前，谁在后”的概念，pytorch官方推荐在token特征后加position encoding；
        #   - 这里的“位置编码”用nn.Parameter，所以是可训练的，会随模型优化一起调整！
        #   - 加法写法符合Transformer的标准实现（即embedding+pos_encoding）

        # ③ pytorch初学者要关注的知识点
        #   - nn.Linear建立的线性层可直接用于(batch_size, feature_num, dim)这样的张量；
        #   - nn.Parameter的张量会自动加进模型参数里，可以被优化；
        #   - src.size(1)得到当前batch的“时间步”数，这样自适应不同输入长度；
        #   - torch的广播（broadcasting）机制保证 pos_encoding（[1,T,hidden_dim]）可直接加到 embedding（[B,T,hidden_dim]）上；
   

        # 第二步：由于 PyTorch 的 nn.Transformer 期望的输入是 (seq_len, batch, feature)
        # 所以这里把 src_emb 从 (batch_size, seq_len, feature_dim)
        # 用 permute 函数转成 (seq_len, batch_size, feature_dim)
        src_emb = src_emb.permute(1, 0, 2)       # (batch, seq, dim) → (seq, batch, dim)

        # 第三步：将输入送入 Transformer 编码器（Encoder部分），
        # 通过多层自注意力捕获时序依赖性与全局表示
        transformer_output = self.transformer.encoder(src_emb)

        # 第四步：编码后的输出再次通过 permute 从 (seq, batch, dim) 转回 (batch, seq, dim)
        # 然后用 view 函数“展平”为 (batch_size, seq_len * hidden_dim)
        # 即将每个时间步得到的特征拼接在一起，准备用于全连接层
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)

        # 第五步：在输出拼接后，为了防止过拟合，这里加一个 Dropout
        transformer_output = self.dropout(transformer_output)

        # 第六步：将经过 Dropout 的特征送入全连接层 self.fc
        # self.fc 的输出维度为 output_dim，即最终的类别数/任务输出数
        predictions = self.fc(transformer_output)

        # 最后：返回 predictions，即模型的最终输出（如类别 logits）
        return predictions
   
```

**流程分解：**

1. **输入**：`src`，形状 `(batch_size, num_timesteps, input_dim)` — 时序特征
2. **嵌入 + 位置编码**：`embedding` 将特征映射到 `hidden_dim`，加上可学习的位置编码
3. **维度变换**：PyTorch 的 `nn.Transformer` 默认期望 `(seq_len, batch, feature)`，所以用 `permute(1, 0, 2)` 转换
4. **编码**：通过 Transformer Encoder 层
5. **展平 + 输出**：将所有时间步的特征拼接（`view`），再通过 Dropout 和 FC 层输出

---

### 2.2 `SAGE` — 图神经网络分支

```26:56:model.py
class SAGE(torch.nn.Module):
    # SAGE模型是PyTorch Geometric (PyG)中常用的图神经网络结构之一，基于GraphSAGE思想。
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        """
        参数说明:
        - in_channels:        输入的节点特征维度
        - hidden_channels:    每层隐藏特征的维度
        - out_channels:       输出特征（通常是最终类别数或图嵌入维度）
        - num_layers:         所用的GCN层数（含输入、输出层）
        - dropout:            dropout概率，防止过拟合

        PyTorch模块结构说明:
        - super(...).__init__():      调用父类nn.Module的初始化（必须）
        - self.convs:                使用ModuleList存储多层SAGEConv（便于for循环多层卷积）
        - SAGEConv:                  图SAGE卷积层（PyG提供），属于消息传递神经网络
        - self.bns:                  BatchNorm1d归一化每层节点特征，防止训练不稳定
        """
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()    # ModuleList: 存放有序的各层卷积
        self.convs.append(SAGEConv(in_channels, hidden_channels))  # 第一层: 输入 -> 隐藏
        self.bns = torch.nn.ModuleList()      # 存放各层的BatchNorm归一化
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))     # 第一层的BN
        # 循环添加中间层 (num_layers-2层): 隐藏->隐藏
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # 最后一层: 隐藏->输出（通常不跟BN和ReLU，只直接输出）
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout  # dropout概率，可调节

    def forward(self, x, edge_index, batch):
        """
        前向传播说明:
        - x:           节点特征 (shape: [num_nodes, in_channels])
        - edge_index:  边索引 (PyG格式，定义图结构)
        - batch:       节点到图的批次映射 (适合小批量多图)
        """
        for i, conv in enumerate(self.convs[:-1]):  # 除最后一层外，其余均relu+dropout
            x = conv(x, edge_index)                 # 图卷积，更新节点特征
            x = self.bns[i](x)                      # BN归一化，提升稳定性
            x = F.relu(x)                           # 非线性激活
            x = F.dropout(x, p=self.dropout, training=self.training)  # dropout
        x = self.convs[-1](x, edge_index)           # 最后一层（输出层）只卷积
        x = global_mean_pool(x, batch)              # 全局均值池化，把所有节点特征聚合成每个图的图向量
        # global_mean_pool是PyG提供的，输入节点特征及batch分组，输出为[batch_size, out_channels]。
        return x
```

**关键设计点：**

- **`ModuleList`**：存储多个卷积层的容器，支持迭代
- **`BatchNorm1d`**：对每层的节点特征做批归一化（因为节点数量可变，不适合 BatchNorm2d）
- **`global_mean_pool`**：PyG（PyTorch Geometric）提供的图级别池化，将所有节点特征取均值，输出图级别嵌入
- **`training` 属性**：`F.dropout` 会根据 `self.training` 自动切换训练/推理行为（训练时 dropout，推理时关闭）

---

### 2.3 `AttentionFusion` — 多模态注意力融合

```182:209:model.py
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        """
        多模态注意力融合层

        参数说明:
        - embed_dim: 每个输入分支的特征维度（embedding size）。必须与MultiheadAttention的输入维数相同。
        - n_heads: 多头注意力的头数。默认1，设为1就是单头注意力，也可设为多头以捕捉不同的交互关系。

        架构理解:
        - 继承了nn.Module，定义了一个深度学习模块
        - self.attn：使用nn.MultiheadAttention实现多头自注意力机制，用于捕捉输入3个分支之间的相互依赖与融合特性，batch_first=True表示x输入的第一个维度是batch size。
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        """
        前向传播流程说明:
        - x：大小为 (batch_size, 3, embed_dim)，即一个batch的3个分支输出，每个分支有embed_dim特征。

        主要步骤详解:
        1. attn(x, x, x)
           - 输入: Query, Key, Value 都用分支堆叠后的x，实现自注意力（Self-Attention）
           - 输出: 
             * attn_output: 融合后的特征，shape为(batch_size, 3, embed_dim)
             * attn_weights: 原始注意力权重，shape为(batch_size, n_heads, 3, 3)
               其中第3、4维分别表示：每个head上分支间相互关注的强度

        2. weights_per_input = attn_weights.mean(dim=1)
           - 多头注意力下对所有head的注意力权重取均值，shape变为(batch_size, 3, 3)
           - 表示每个分支给所有分支的attention平均贡献

        3. contribution_per_input = weights_per_input.sum(dim=1)
           - 沿输入分支维累加，相当于统计每个分支整体被关注的程度，shape为(batch_size, 3)

        4. fused = attn_output.reshape(attn_output.size(0), -1)
           - 将3个分支的特征拼接成一个长特征向量 (batch_size, 3*embed_dim)

        返回:
        - 融合后的特征fused（可用于下游分类等任务）
        - 分支贡献度矩阵weights_per_input
        """
        # x: (batch_size, 3, embed_dim) — 3个分支的输出
        attn_output, attn_weights = self.attn(x, x, x)
        # attn_output: (batch_size, 3, embed_dim)
        # attn_weights: (batch_size, num_heads, 3, 3)
        weights_per_input = attn_weights.mean(dim=1)  # (batch_size, 3, 3)
        contribution_per_input = weights_per_input.sum(dim=1)  # (batch_size, 3)
        fused = attn_output.reshape(attn_output.size(0), -1)  # 展平为 (batch_size, 3*embed_dim)
        return fused, weights_per_input

**注意力融合机制：**

- 将三个分支的输出作为三个"token"，用 **Self-Attention** 让它们互相交互
- 每个分支的特征既是 Query 也是 Key 和 Value（self-attention）
- `attn_weights` 揭示了每个分支对最终融合结果的贡献权重
- `reshape(-1)` 将三个 embed_dim 拼接成一个长向量

---

### 2.4 `T3Former` — 三分支融合主模型

```239:289:model.py
class T3Former(nn.Module):
    """
    T3Former 是一个三分支（多模态）融合的时序图分类主模型。它将三个不同模态的分支输出，通过自注意力机制进行融合，最终用于分类任务。

    __init__ 的每个参数依次解析：
    - sage_input_dim: 图结构分支（GraphSAGE）输入特征的维度，通常等于每个节点的原始特征长度。
    - transformer_input_dim: 用于Betti特征分支的Transformer的输入维度，即每个时间步Betti数特征的长度。
    - transformer2_input_dim: 用于DoS特征分支的Transformer输入维度，即每个时间步DoS（Density of States）特征的长度。
    - hidden_dim: Transformer各分支的隐藏层维度，也是特征变换过程中各层的通道数或隐表示维度。
    - output_dim: 最终分类输出的类别数（如二分类为2，多分类为N）。
    - n_heads: Transformer编码器的多头注意力头数；决定每层并行注意力的独立子空间数，有助于学习多样化的关系。
    - n_layers: Transformer编码器的堆叠层数，越深表达能力越强，但也更耗算力。
    - num_timesteps1/2: Betti分支和DoS分支各自时序特征的时间窗口长度（通常决定输入的序列长度）。
    - dropout_p: Dropout概率，用于防止过拟合。
    - final_output_dim: 每个分支最终导出用于融合的特征维度，默认10。即每个分支输出长度为final_output_dim的向量。

    结构设计意义：
    - 三个分支（transformer_branch/transformer_branch2/sage_branch）分别针对不同比特征模态建模，充分提取时序拓扑（Betti）、时序谱密度（DoS）、空间结构（图结构）信息。
    - attn_fusion 层用 MultiheadAttention，通过自注意力机制，将三个分支的输出作为三个token在特征层面进行信息交互/加权融合，实现多模态特征集成。
    - fc_final 是最后的全连接层，把self-attn输出的拼接特征（长度3*final_output_dim）变换到分类输出空间。
    - dropout 层提升模型泛化能力。

    格式说明、逐行注释：

    """
    def __init__(self,
                 sage_input_dim, transformer_input_dim, transformer2_input_dim,
                 hidden_dim, output_dim, n_heads, n_layers,
                 num_timesteps1, num_timesteps2, dropout_p,
                 final_output_dim=10):
        super().__init__()

        # 三大分支
        # -- Betti特征分支，建模拓扑同调序列
        self.transformer_branch = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=hidden_dim,
            num_classes=final_output_dim,
            n_heads=n_heads,
            num_layers=n_layers,
            num_timesteps=num_timesteps1,
            dropout_p=dropout_p
        )
        # -- DoS特征分支，建模谱密度序列
        self.transformer_branch2 = TransformerClassifier(
            input_dim=transformer2_input_dim,
            hidden_dim=hidden_dim,
            num_classes=final_output_dim,
            n_heads=n_heads,
            num_layers=n_layers,
            num_timesteps=num_timesteps2,
            dropout_p=dropout_p
        )
        # -- 图结构分支，建模空间关系（节点/边特征）
        self.sage_branch = SAGE(
            in_channels=sage_input_dim,
            hidden_dim=hidden_dim,
            out_dim=final_output_dim,
            dropout_p=dropout_p
        )

        # 注意力融合（将三分支输出视为3个token做multihead self-attention）
        self.attn_fusion = AttentionFusion(embed_dim=final_output_dim, n_heads=1)
        # 输出前Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        # 将融合后特征映射到输出类别数，全连接（输入3*final_output_dim，输出output_dim）
        self.fc_final = nn.Linear(3 * final_output_dim, output_dim)

    def forward(self, x_gsage, edge_index, batch, transformer_input, transformer_input1):
        """
        - x_gsage: 图结构分支的节点特征输入，shape=(B, num_nodes, sage_input_dim)
        - edge_index: 边连接关系（PyG规范）
        - batch: 每个节点属于哪个图（PyG mini-batch）
        - transformer_input: Betti特征序列输入，shape=(B, num_timesteps1, transformer_input_dim)
        - transformer_input1: DoS特征序列输入，shape=(B, num_timesteps2, transformer2_input_dim)
        """
        # 1. 三分支独立计算特征
        out1 = self.transformer_branch(transformer_input)      # Betti分支，输出(B, final_output_dim)
        out2 = self.transformer_branch2(transformer_input1)    # DoS分支，输出(B, final_output_dim)
        out3 = self.sage_branch(x_gsage, edge_index, batch)    # 图结构分支，输出(B, final_output_dim)

        # 2. 将三分支输出沿新维度堆叠，(B, 3, final_output_dim)
        stacked = torch.stack([out1, out2, out3], dim=1)
        # stack重点：数据shape从(B, final_output_dim) -> (B, 3, final_output_dim)，3作为“token数量”，便于multi-head attn。
        
        # 3. 三分支特征用注意力融合，得到加权融合特征(B, 3*final_output_dim)和分支权重
        attn_out, weights = self.attn_fusion(stacked)           # attn_out: (B, 3*final_output_dim)
        
        # 4. Dropout + 全连接，输出分类结果
        attn_out = self.dropout(attn_out)
        output = self.fc_final(attn_out)
        return output, weights

# 总结：
# - 参数命名体现了跨模态/多分支输入对齐
# - 模型结构高度模块化，利于多模态时序图的拓扑-谱-结构联动表征
```

**`torch.stack` vs `torch.cat`**：

```python
# torch.stack: 在新维度上叠加 → 形状 (B, 3, D)
stacked = torch.stack([out1, out2, out3], dim=1)

# torch.cat: 在已有维度上拼接 → 形状 (B, 3*D)
concatenated = torch.cat([out1, out2, out3], dim=1)
```

`stack` 增加一个新维度，`cat` 只拼接现有维度——T3Former 用 `stack` 是为了让 AttentionFusion 把 3 个分支当作 3 个 token 来处理。

---

### 2.5 `CNNTransformer` — CNN + Transformer 分支

```59:101:model.py
class CNNTransformer(nn.Module):
    def __init__(self, num_classes, cnn_channels=64, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # CNN 特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU()
        )

        # 将空间网格展平成 patch 序列
        self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N, C)

        self.embedding = nn.Linear(cnn_channels, d_model)  # 投影到 d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))      # 可学习的 [CLS] token

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 4, 20, 10) — 4通道，20×10 网格
        x = self.cnn(x)                                          # (B, 64, 20, 10)
        x = self.flatten_patches(x)                              # (B, 200, 64)
        x = self.embedding(x)                                   # (B, 200, d_model)

        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)                    # (B, 201, d_model)

        x = self.transformer(x)                                  # (B, 201, d_model)
        cls_output = x[:, 0]                                     # 取 [CLS] token
        return self.classifier(cls_output)
```

**`nn.TransformerEncoderLayer` vs `nn.Transformer`**：

| | `nn.Transformer` | `nn.TransformerEncoderLayer` |
|---|---|---|
| 组成 | 完整的 Encoder + Decoder | 单个 Encoder 层 |
| 用途 | 需要 Encoder+Decoder 的任务 | 只需要 Encoder（如分类） |
| 组合 | 配合 `nn.TransformerEncoder` 使用 | 单独使用 |

这里 `nn.TransformerEncoderLayer` + `nn.TransformerEncoder` = 只用 Encoder 部分，功能等价于 `nn.Transformer(d_model=..., ...).encoder`。

**`expand` 与共享参数**：

```python
cls_token = self.cls_token.expand(B, -1, -1)
```

`self.cls_token` 形状 `(1, 1, d_model)`，`.expand(B, -1, -1)` 创建了形状 `(B, 1, d_model)` 的视图，**不复制内存**——所有 batch 样本共享同一个 [CLS] token 的可学习参数，符合 ViT/Vision Transformer 的标准做法。

---

### 2.6 `T3SAGE` — 简化版融合（T3Former 的无注意力版本）

```291:341:model.py
class T3SAGE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 分支结构与 T3Former 完全相同
        self.transformer_branch = TransformerClassifier(...)
        self.transformer_branch2 = TransformerClassifier(...)
        self.sage_branch = SAGE(...)

        # 区别：使用直接拼接代替注意力融合
        combined_feature_dim = 3 * final_output_dim
        self.attn_fusion = AttentionFusion(embed_dim=final_output_dim, n_heads=1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_final = nn.Linear(combined_feature_dim, output_dim)

    def forward(self, x_gsage, edge_index, batch, transformer_input, transformer_input1):
        out1 = self.transformer_branch(transformer_input)
        out2 = self.transformer_branch2(transformer_input1)
        out3 = self.sage_branch(x_gsage, edge_index, batch)

        # 直接拼接，而非注意力融合
        combined = torch.cat([out1, out2, out3], dim=1)  # (B, 3*final_dim)
        combined = self.dropout(combined)
        output = self.fc_final(combined)
        return output  # 不返回 weights
```

**T3SAGE vs T3Former**：

| | T3Former | T3SAGE |
|---|---|---|
| 融合方式 | AttentionFusion（自注意力加权） | 直接 `torch.cat` 拼接 |
| 返回值 | `output, weights` | `output` |
| 参数量 | 更多（AttentionFusion 层） | 更少 |
| 可解释性 | `weights` 揭示各分支贡献 | 无权重输出 |

---

### 2.7 `T3GNN` — 交叉注意力变体

```361:410:model.py
class T3GNN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.sage_branch = SAGE(...)

        # Betti 和 DoS 特征先投影到统一维度
        self.betti_proj = nn.Linear(4, final_output_dim)
        self.dos_proj = nn.Linear(4, final_output_dim)

        # 交叉注意力
        self.cross_attn = CrossAttention(final_output_dim, n_heads=1)

        self.transformer = TransformerClassifier(...)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, data_seq, betti_seq, dos_seq):
        # data_seq: 图的时序列表，betti_seq: (B, T, 4)，dos_seq: (B, T, 4)
        graph_embeds = []
        for t in range(len(data_seq)):
            batch_t = Batch.from_data_list(data_seq[t])
            out = self.sage_branch(batch_t.x, batch_t.edge_index, batch_t.batch)
            graph_embeds.append(out)

        graph_embeds = torch.stack(graph_embeds, dim=1)  # (B, T, final_dim)
        betti_embeds = self.betti_proj(betti_seq)         # (B, T, final_dim)
        dos_embeds = self.dos_proj(dos_seq)              # (B, T, final_dim)

        # 交叉注意力：query=GNN，key/value=Betti+DoS
        fused = self.cross_attn(graph_embeds, betti_embeds, dos_embeds)
        fused = self.dropout(fused)
        output = self.transformer(fused)
        return output
```

**T3GNN 的设计创新**：

- 与 T3Former 的"三分支并行 → 融合"不同，T3GNN 是**顺序交叉**的：
  1. 图结构通过 SAGE 生成时序嵌入
  2. Betti 和 DoS 投影到同维度后，以 GNN 输出为 Query 去查询它们
  3. 融合结果再通过 Transformer 编码时序依赖

---

### 2.8 `CrossAttention` — 交叉注意力模块

```347:356:model.py
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, query, key_value, key2):
        # query: (batch, seq_len, embed_dim) 来自 SAGE
        # key_value: (batch, seq_len, embed_dim) 来自 Betti
        # key2: (batch, seq_len, embed_dim) 来自 DoS
        out, _ = self.attn(query, key_value, key2)
        return out
```

**CrossAttention vs Self-Attention**：

| | Self-Attention (AttentionFusion) | Cross-Attention (CrossAttention) |
|---|---|---|
| Q/K/V 来源 | 同一个输入 | 不同的输入 |
| 用法 | 三分支互相"交流"找权重 | SAGE 特征"查询"拓扑特征 |
| 典型场景 | 特征融合 | 跨模态对齐 |

---

### 2.9 `GNN` — 通用 GNN（支持 8 种架构）

```415:459:model.py
class GNN(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        elif model_type == 'GIN':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)  # GIN 需要自定义邻居聚合逻辑
            self.conv2 = GINConv(nn2)
        elif model_type == 'TAG':
            self.conv1 = TAGConv(in_channels, hidden_channels)
            self.conv2 = TAGConv(hidden_channels, hidden_channels)
        elif model_type == 'Cheb':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=3)  # K阶切比雪夫多项式
            self.conv2 = ChebConv(hidden_channels, hidden_channels, K=3)
        elif model_type == 'ARMA':
            self.conv1 = ARMAConv(in_channels, hidden_channels)
            self.conv2 = ARMAConv(hidden_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, num_classes)
```

**6 种 GNN 架构对比**：

| 架构 | 核心思想 | 聚合方式 | 特点 |
|---|---|---|---|
| **GCN** | 谱域卷积近似 | 归一化邻居平均 | 最基础，效率高 |
| **GraphSAGE** | 归纳式学习 | 采样 + 多种聚合器 | 支持大图、可泛化 |
| **GAT** | 注意力加权 | 加权邻居聚合 | 注意力可解释 |
| **GIN** | 同构性最强 | sum 聚合 + MLP | 理论表达能力最强 |
| **TAG** | 吸收 GCN | 一阶邻居 + 二阶邻居 | 兼顾局部与全局 |
| **ChebNet** | 切比雪夫多项式 | K阶多项式近似谱卷积 | 无需计算拉普拉斯矩阵特征分解 |
| **ARMA** | ARMA 滤波器 | 多层 ARMA 堆叠 | 更稳定的频域响应 |

---

### 2.10 `GraphTransformer` — 图 Transformer

```462:474:model.py
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)       # 节点级别 Transformer
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)
```

**`TransformerConv`（PyG）vs `nn.MultiheadAttention`**：

| | `TransformerConv`（PyG） | `nn.MultiheadAttention` |
|---|---|---|
| 输入 | 节点特征 + 边索引 | Q/K/V 张量 |
| 位置编码 | 内置（可学） | 需手动加 |
| 稀疏性 | 稀疏（只算邻居） | 全连接 |
| 应用场景 | 图神经网络 | Transformer/NLP |

---

### 2.11 `GPSModel` — 混合 GNN + 全注意力

```524:547:model.py
class GPSModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        # GPSConv = 局部 GNN 卷积 + 全注意力机制的混合
        self.conv1 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.conv2 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)
```

**`GPSConv` 的设计哲学**：

`GPSConv = 局部图卷积（GCN）+ 全局自注意力`

- GCN 部分捕获局部拓扑结构（邻居信息）
- 自注意力部分捕获全局关系（所有节点交互）
- 两者结合，兼顾局部和全局建模能力

---

### 2.12 `Graphormer` — Graphormer 变体

```551:583:model.py
class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, heads=4, max_degree=10):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)

        # 度编码：每个节点的度数映射为嵌入向量
        self.degree_emb = Embedding(max_degree + 1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
            )

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, deg=None):
        x = self.input_proj(x)

        if deg is not None:
            deg = deg.clamp(max=self.degree_emb.num_embeddings - 1)
            x = x + self.degree_emb(deg)  # 度编码注入结构信息

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x + residual)  # 残差连接
        x = global_mean_pool(x, batch)
        return self.classifier(x)
```

**`nn.Embedding` — 离散 ID 到向量的映射**：

```python
self.degree_emb = Embedding(max_degree + 1, hidden_channels)
x = x + self.degree_emb(deg)  # deg 是度的整数值，映射为向量
```

`nn.Embedding(num_embeddings, embedding_dim)` 维护一个可学习的查找表（形状 `num_embeddings × embedding_dim`），输入整数索引，输出对应的嵌入向量——常用于词嵌入、度编码等离散特征。

**残差连接（Residual Connection）**：

```python
x = norm(x + residual)
```

`norm(x + residual)` 让梯度可以直接回传，训练更稳定，是现代深度网络的标配（ResNet 核心思想）。

---

### 2.13 `NeuroGraphDynamic` — 神经影像动态图数据集

```15:78:NeuroGraph.py
class NeuroGraphDynamic():
    url = 'https://vanderbilt.box.com/shared/static/...'
    filenames = {
        'DynHCPGender': 'mj0z6unea34lfz1hkdwsinj7g22yohxn.zip',
        'DynHCPActivity': '2so3fnfqakeu6hktz322o3nm2c8ocus7.zip',
        'DynHCPAge': '195f9teg4t4apn6kl6hbc4ib4g9addtq.zip',
        'DynHCPWM': 'mxy8fq3ghm60q6h7uhnu80pgvfxs6xo2.zip',
        'DynHCPFI': 'un7w3ohb2mmyjqt1ou2wm3g87y1lfuuo.zip',
    }

    def __init__(self, root, name):
        self.root = root
        self.name = name
        assert name in self.filenames.keys()
        file_path = os.path.join(self.root, self.name, 'processed', self.name + ".pt")
        if not os.path.exists(file_path):
            self.download()
        self.dataset, self.labels = self.load_data()
```

**注意**：`NeuroGraphDynamic` **不是** `nn.Module` 的子类，而是一个普通 Python 类（负责数据下载和加载），与 `torch.utils.data.Dataset` 同类。

**数据集说明**：

| 数据集 | 领域 | 任务 |
|---|---|---|
| DynHCPGender | 神经影像 | 性别分类 |
| DynHCPActivity | 神经影像 | 活动分类 |
| DynHCPAge | 神经影像 | 年龄预测 |
| DynHCPWM / DynHCPFI | 神经影像 | 工作记忆 / 流体智力 |

---

### 2.14 Betti / DoS 特征提取 — 拓扑特征工程

#### DoS（Density of States，态密度）特征

```29:41:dos_betti_features_social.py
def compute_dos(graph, num_bins=4, bandwidth=0.05):
    if graph.number_of_nodes() == 0:
        return np.zeros(num_bins)
    # 归一化拉普拉斯矩阵的特征值
    L = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues = eigh(L, eigvals_only=True)
    # 核密度估计，将特征值分布离散化为 num_bins 个 bins
    kde = gaussian_kde(eigenvalues, bw_method=bandwidth)
    x_min, x_max = min(eigenvalues), max(eigenvalues)
    bin_centers = np.linspace(x_min, x_max, num_bins)
    dos_values = kde(bin_centers)
    return dos_values
```

**DoS 的物理含义**：图拉普拉斯矩阵的特征值分布（光谱）反映图的连通性结构——DoS 将连续的光谱离散化为 4 个 bins 的概率密度值，作为拓扑特征输入 Transformer。

#### Betti 数特征（拓扑不变量）

```60:84:dos_betti_features_neuro.py
def betti_extraction(data):
    ptr = data.ptr
    edge_index = data.edge_index
    results = []
    for j in range(len(ptr) - 1):
        node_start, node_end = ptr[j].item(), ptr[j + 1].item()
        mask = ((edge_index[0] >= node_start) & (edge_index[0] < node_end))
        edges = edge_index[:, mask] - node_start
        G = nx.Graph()
        G.add_nodes_from(range(node_end - node_start))
        G.add_edges_from(edges.t().tolist())
        adj_matrix = nx.to_numpy_array(G)
        # 用 pyflagser 计算持久同调 Betti 数
        homology = pyflagser.flagser_unweighted(
            adj_matrix, min_dimension=0, max_dimension=2, directed=False, coeff=2
        )
        b0, b1 = homology['betti'][0], homology['betti'][1]
        results.append((b0, b1, G.number_of_nodes(), G.number_of_edges()))
    return results
```

**Betti 数的拓扑含义**：

| Betti 数 | 含义 | 对应拓扑特征 |
|---|---|---|
| β₀ (b0) | 连通分量数 | 图中独立组件的数量 |
| β₁ (b1) | 一维洞数 | 环/循环的数量 |
| β₂ (b2) | 二维空洞数 | 空腔/空洞的数量 |

#### 滑动窗口时序特征

```43:66:dos_betti_features_social.py
def sliding_window_dos(G, thresholds, window=3, jump=1, num_bins=4):
    results = []
    num_windows = (len(thresholds) - window) // jump + 1
    for i in range(0, num_windows * jump, jump):
        t_start = thresholds[i]
        t_end = thresholds[i + window - 1]
        # 过滤出窗口内的活跃边，构建子图
        active_edges = [
            (u, v) for u, v, attrs in edges_data
            if any(t_start <= t <= t_end for t in attrs['time'])
        ]
        subgraph = nx.Graph()
        subgraph.add_edges_from(active_edges)
        dos_vector = compute_dos(subgraph, num_bins=num_bins)
        results.append(dos_vector)
    return np.stack(results)
```

**滑动窗口的作用**：将连续的时序图切分为多个时间窗口，在每个窗口内计算拓扑特征（DoS 或 Betti），生成时序特征向量序列，送入 Transformer 编码时序依赖。

---

### 2.15 `train_GNN.py` — GNN 模型对比实验

```5:16:train_GNN.py
from model import GNN, GraphTransformer, GPSModel, Graphormer

models_to_run = ['SAGE', 'GAT', 'UniMP', 'graphormer'] if args.model == 'all' else [args.model]
for model_name in models_to_run:
    if model_name in ['GCN', 'SAGE', 'GAT', 'GIN', 'TAG']:
        model = GNN(model_name, in_channels=data_list[0].x.size(1),
                    hidden_channels=args.hidden_dim, num_classes=num_class).to(device)
    elif model_name == 'UniMP':
        model = GraphTransformer(...)  # 即论文中的 UniMP（统一消息传递注意力）
    elif model_name == 'graphormer':
        model = Graphormer(...)
    elif model_name == 'GPS':
        model = GPSModel(...)
```

**支持的模型列表**：

| 名称 | 模型类 | 备注 |
|---|---|---|
| GCN / SAGE / GAT / GIN / TAG | `GNN` | 通过 `model_type` 参数切换 |
| UniMP | `GraphTransformer` | 即 Unified Message Passing Attention |
| graphormer | `Graphormer` | 带度编码的 GAT 变体 |
| GPS | `GPSModel` | 局部 GNN + 全注意力混合 |

---

## 3. PyTorch 核心概念详解（代码对照）

### 3.1 `torch.nn.Module` — 一切模型的基类

```python
class TransformerClassifier(nn.Module):
    def __init__(self, ...):
        super(TransformerClassifier, self).__init__()
```

**PyTorch 规则：所有神经网络层必须继承 `nn.Module`。**

- `super().__init__()` 激活 Module 的初始化逻辑（注册子模块、参数等）
- 只有通过 `super().__init__()` 注册的层，PyTorch 才能自动追踪参数和调用 `.to(device)`、`.parameters()` 等方法

---

### 3.2 `super().__init__()` 与模块注册

当你写 `self.embedding = nn.Linear(...)` 时，PyTorch 会自动：

1. 将该层加入模块的内部注册表
2. 自动包含在 `model.parameters()` 中
3. 自动支持 `.to(device)` 移动到 GPU

**对比错误写法（不会自动注册）：**

```python
# ❌ 错误：局部变量，PyTorch 不知道它的存在
embedding = nn.Linear(input_dim, hidden_dim)

# ✅ 正确：实例属性，PyTorch 会自动注册
self.embedding = nn.Linear(input_dim, hidden_dim)
```

---

### 3.3 `forward()` — 数据流向

`forward()` 是**必须定义**的核心方法，定义了模型如何将输入变成输出：

```python
def forward(self, src):
    src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
```

- 前向传播的逻辑完全由你决定
- `model(x)` 会自动调用 `model.forward(x)`
- `backward()` 不需要手动写，PyTorch 的 `autograd` 会自动根据 `forward()` 计算图推导梯度

---

### 3.4 `nn.Parameter` — 可学习参数

```python
self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
```

**什么是 Parameter？**

- `nn.Parameter` 是 `torch.Tensor` 的子类
- PyTorch 自动将所有 `Parameter` 加入 `model.parameters()`，让优化器可以更新它们
- `torch.zeros(...)` 创建全零张量作为初始值，后续通过训练学习最优的位置编码

**与普通 Tensor 的区别：**

```python
# 普通 Tensor — 不会自动作为模型参数
x = torch.zeros(10)

# nn.Parameter — 会被优化器更新
x = nn.Parameter(torch.zeros(10))
```

---

### 3.5 `ModuleList` 与 `ModuleDict`

```python
self.convs = torch.nn.ModuleList()
self.convs.append(SAGEConv(in_channels, hidden_channels))
```

**为什么用 ModuleList 而不用 Python 列表？**

```python
# ❌ Python 列表 — PyTorch 不知道这些层
convs = [SAGEConv(...), SAGEConv(...), SAGEConv(...)]

# ✅ ModuleList — PyTorch 正确注册，可以迭代
convs = torch.nn.ModuleList([SAGEConv(...), SAGEConv(...), SAGEConv(...)])
```

`ModuleList` 使得循环定义多层变得优雅，且 PyTorch 能正确追踪所有子模块的参数。

---

### 3.6 常用层速查

| 层 | 代码 | 说明 |
|---|---|---|
| **线性变换** | `nn.Linear(in, out)` | `y = xW^T + b`，权重自动初始化 |
| **Dropout** | `nn.Dropout(p=0.5)` | 训练时随机置零，推理时关闭 |
| **ReLU** | `F.relu(x)` 或 `nn.ReLU()` | 激活函数，in-place 版本 `F.relu(x, inplace=True)` |
| **BatchNorm1d** | `nn.BatchNorm1d(channels)` | 对通道维度做归一化，适用于 GNN 输出的节点特征 |
| **LayerNorm** | `nn.LayerNorm(normalized_shape)` | 对最后一个维度做归一化，适用于 Transformer 输出 |
| **Embedding** | `nn.Embedding(num, dim)` | 整数索引 → 固定维度向量，常用于词嵌入、度编码 |
| **Sequential** | `nn.Sequential(...)` | 按顺序执行子模块，适合简单堆叠 |

### 3.6.1 `nn.Embedding` — 离散索引查表

```python
# Graphormer 中的度编码
self.degree_emb = Embedding(max_degree + 1, hidden_channels)
x = x + self.degree_emb(deg)  # deg: (batch_size,)，输出: (batch_size, hidden_dim)
```

**原理**：维护一个形状为 `(num_embeddings, embedding_dim)` 的可学习查找表，输入整数 `i`，返回第 `i` 行向量。

**与 `nn.Linear` 的区别**：

| | `nn.Embedding` | `nn.Linear` |
|---|---|---|
| 输入 | 整数索引（离散） | 连续向量 |
| 权重形状 | `(num_embeddings, embed_dim)` | `(out_features, in_features)` |
| 典型用途 | 词嵌入、度编码 | 特征投影 |

### 3.6.2 残差连接（Residual Connection）

```python
# Graphormer 中的残差连接
for conv, norm in zip(self.layers, self.norms):
    residual = x
    x = conv(x, edge_index)
    x = F.relu(x)
    x = norm(x + residual)  # 残差：直接将输入加到输出上
```

**为什么需要残差连接？**

- 深层网络中梯度消失会导致训练困难
- `x + residual` 让梯度可以直接回传，训练更稳定
- ResNet（CV）和 Transformer（NLP）都是残差连接的典型受益者

### 3.6.3 `tensor.expand()` — 零成本维度扩展

```python
cls_token = self.cls_token.expand(B, -1, -1)  # (1,1,d) → (B,1,d)
```

**特点**：

- `expand` **不复制数据**，只是改变张量的"视图"（stride）
- 所有 batch 样本共享同一个 [CLS] token 参数
- `-1` 表示该维度保持不变
- 与 `torch.tile`（会复制数据）形成对比

### 3.6.4 `zip` + `ModuleList` 配对迭代

```python
for conv, norm in zip(self.layers, self.norms):
    x = norm(conv(x) + x)
```

**技巧**：`zip` 将两个等长的 ModuleList 配对，用一个循环同时遍历各层的卷积和归一化——比下标索引更 Pythonic。

---

### 3.7 `nn.Transformer` — PyTorch 内置 Transformer

```python
self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, 
                                   num_encoder_layers=n_layers, 
                                   num_decoder_layers=n_layers)
```

**核心参数：**

| 参数 | 含义 |
|---|---|
| `d_model` | 输入/输出的特征维度 |
| `nhead` | 注意力头数（`d_model` 必须能被 `nhead` 整除） |
| `num_encoder_layers` | Encoder 层数 |
| `num_decoder_layers` | Decoder 层数 |

**数据维度约定：**

```python
# nn.Transformer 默认期望 (seq_len, batch, d_model)
src_emb = src_emb.permute(1, 0, 2)  # (batch, seq, d_model) → (seq, batch, d_model)
output = self.transformer.encoder(src_emb)  # 输出: (seq, batch, d_model)
```

---

### 3.8 `nn.MultiheadAttention`

```python
self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)
attn_output, attn_weights = self.attn(x, x, x)
```

- **`batch_first=True`**：输入/输出格式为 `(batch, seq, feature)`（更直观）
- **返回值**：`attn_output` 是注意力加权后的输出，`attn_weights` 是注意力分数矩阵
- **在这里的用法**：三分支输出拼成 3 个 token，做 self-attention，让模型学习每个分支的重要性权重

---

### 3.9 `nn.Sequential`

```python
self.cnn = nn.Sequential(
    nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(cnn_channels),
    nn.ReLU(),
    nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(cnn_channels),
    nn.ReLU()
)
```

**`nn.Sequential` 的特点：**

- 按顺序依次执行每一层
- 内部自动实现 `forward()`，无需手动编写
- 适合简单的线性堆叠结构
- 比手动写 `forward` 更简洁

---

## 4. 训练流程解析

### 4.1 完整训练循环（`train_T3Former.py`）

```python
# 1. 数据加载
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate)

# 2. 模型初始化
model = T3Former(...).to(device)

# 3. 优化器与损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 4. 训练循环
for epoch in range(1, args.epochs + 1):
    model.train()                        # 切换到训练模式（激活 dropout 等）
    for xb, xb1, graph_batch, yb in train_loader:
        # 数据移动到 GPU
        graph_batch = graph_batch.to(device)
        xb = xb.to(device)
        xb1 = xb1.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()             # 清除上一步的梯度
        output, weights = model(...)      # 前向传播
        loss = criterion(output, yb)     # 计算损失
        loss.backward()                  # 反向传播（自动计算梯度）
        optimizer.step()                 # 更新参数

    # 验证
    model.eval()                         # 切换到推理模式
    with torch.no_grad():                # 禁用梯度计算，节省显存
        for xb, xb1, graph_batch, yb in val_loader:
            output, _ = model(...)
```

### 4.2 PyTorch 训练三步曲

```
optimizer.zero_grad()   ──→  loss.backward()   ──→  optimizer.step()
  清除旧梯度                   自动求导                  更新参数
```

**重要警告**：如果忘记 `zero_grad()`，梯度会累积，导致参数更新错误。

### 4.3 `model.train()` vs `model.eval()`

| 方法 | 效果 |
|---|---|
| `model.train()` | Dropout 生效，BatchNorm 用当前 batch 的统计量 |
| `model.eval()` | Dropout 关闭，BatchNorm 用训练好的全局统计量 |

### 4.4 `torch.no_grad()`

```python
with torch.no_grad():
    output = model(x)
```

- 禁用梯度计算和存储，大幅减少显存占用
- 验证/测试阶段必须使用
- 不影响模型参数的梯度（模型本身不在 `no_grad` 下时仍可训练）

---

## 5. 数据处理管线

### 5.1 自定义 Dataset

```python
class CustomGraphDataset(Dataset):
    def __init__(self, X, X1, graphs, y):
        self.X = X       # Betti 特征
        self.X1 = X1     # DoS 特征
        self.graphs = graphs  # 图数据
        self.y = y       # 标签

    def __len__(self):
        return len(self.y)  # 数据集大小

    def __getitem__(self, idx):
        # 返回一个样本
        return self.X[idx], self.X1[idx], self.graphs[idx], self.y[idx]
```

**Dataset 三大必须实现的方法：**

| 方法 | 作用 |
|---|---|
| `__init__` | 初始化，存储数据 |
| `__len__` | 返回数据集样本数量 |
| `__getitem__` | 根据索引返回单个样本 |

### 5.2 自定义 collate_fn

```python
def custom_collate(batch):
    X_batch, X1_batch, graph_batch, y_batch = zip(*batch)
    return (
        torch.stack(X_batch),              # (N, T, F)
        torch.stack(X1_batch),              # (N, T, F)
        Batch.from_data_list(graph_batch), # PyG Batch（合并多个图为一个大图）
        torch.stack(y_batch)               # (N,)
    )
```

**为什么需要 collate_fn？**

- DataLoader 默认只能处理形状统一的 tensor
- 图数据的节点数量不同，无法直接 batch，需要 `Batch.from_data_list()` 将多个图合并为一个批量图，同时用 `batch` 向量标记每个节点属于哪个原始图

### 5.3 数据增强与归一化

```python
scaler = MinMaxScaler()
for i in range(len(data_list)):
    x_scaled = scaler.fit_transform(data_list[i].x)
    data_list[i].x = x_scaled
```

使用 sklearn 的 `MinMaxScaler` 对节点特征做归一化到 `[0, 1]` 区间。

---

## 6. 架构总结图

```
┌─────────────────────────────────────────────────────────────┐
│                         输入层                               │
│  Betti特征 (T×4)   DoS特征 (T×4)      图结构 (节点+边)       │
└───────┬──────────────────┬──────────────────────┬──────────┘
        │                  │                      │
        ▼                  ▼                      ▼
┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐
│  Linear嵌入   │  │   Linear嵌入    │  │    SAGEConv层×N    │
│ + 位置编码    │  │  + 位置编码     │  │  + BatchNorm       │
└───────┬───────┘  └────────┬────────┘  └─────────┬──────────┘
        │                   │                      │
        ▼                   ▼                      ▼
┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐
│ nn.Transformer│  │nn.Transformer   │  │ global_mean_pool   │
│  Encoder      │  │  Encoder        │  │ (节点→图嵌入)      │
└───────┬───────┘  └────────┬────────┘  └─────────┬──────────┘
        │                   │                      │
        ▼                   ▼                      ▼
  (B, 10)           (B, 10)             (B, 10)
        │                   │                      │
        └────────┬──────────┴──────────────────────┘
                 │ torch.stack → (B, 3, 10)
                 ▼
        ┌─────────────────────┐
        │   AttentionFusion  │  ← nn.MultiheadAttention
        │  (自注意力融合)     │
        └────────┬────────────┘
                 │ (B, 30)
                 ▼
        ┌─────────────────────┐
        │  Dropout + FC       │
        │  Linear(30 → num_classes)
        └────────┬────────────┘
                 ▼
           (B, num_classes)  →  分类结果
```

---

## 附录：关键 PyTorch API 速查表

| API | 作用 |
|---|---|
| `torch.device('cuda:0')` | 指定计算设备 |
| `model.to(device)` | 将模型参数移动到 GPU |
| `optimizer.zero_grad()` | 清除梯度 |
| `loss.backward()` | 反向传播 |
| `optimizer.step()` | 更新参数 |
| `model.train()` / `model.eval()` | 切换训练/推理模式 |
| `torch.no_grad()` | 推理时禁用梯度 |
| `nn.ModuleList` | 存储多个子模块 |
| `nn.ModuleDict` | 键值对存储子模块 |
| `nn.Parameter` | 可学习参数 |
| `nn.Sequential` | 顺序容器 |
| `nn.Embedding(num, dim)` | 整数索引 → 向量查表 |
| `nn.LayerNorm` | Layer Normalization |
| `nn.BatchNorm1d` | Batch Normalization |
| `torch.stack` | 沿新维度拼接张量 |
| `torch.cat` | 沿已有维度拼接张量 |
| `tensor.permute(d0,d1,...)` | 维度重排 |
| `tensor.view(shape)` | 改变形状（需连续） |
| `tensor.contiguous()` | 确保内存连续 |
| `tensor.expand(shape)` | 零成本扩展维度 |
| `tensor.clamp(max=...)` | 限制上限值 |
| `Batch.from_data_list()` | PyG 中将多个 Data 合并为 Batch |
| `global_mean_pool` | 图级别池化（PyG） |
| `SAGEConv / GCNConv / GATConv` | PyG 图卷积层 |
| `TransformerConv` | PyG Transformer 图卷积 |
| `GPSConv` | 混合 GNN + 全注意力（PyG） |
| `eigh()` | scipy 矩阵特征分解（计算拉普拉斯谱） |
| `gaussian_kde()` | scipy 核密度估计 |
| `pyflagser.flagser_unweighted()` | 计算持久同调 Betti 数 |
