# PyTorch 内置方法详解

本文件详细解释 Level 1 练习中用到的所有 PyTorch 内置方法。

---

## 目录

1. [张量创建方法](#1-张量创建方法)
2. [数学运算方法](#2-数学运算方法)
3. [张量变换方法](#3-张量变换方法)
4. [神经网络层](#4-神经网络层)
5. [其他常用方法](#5-其他常用方法)

---

## 1. 张量创建方法

### `torch.randn()`

**作用**：创建指定形状的随机张量（标准正态分布）

```python
import torch

# 语法
torch.randn(*size)

# 示例
Q = torch.randn(2, 3, 4)  # 创建 [2, 3, 4] 的随机张量
# 输出: tensor([[[ 0.1048, -0.3172,  1.3286, -0.2696],
#               [ 0.0962,  0.5200, -0.8936,  0.9244],
#               [-0.1273, -1.1998,  0.6207, -0.7147]],
#              [[-0.8948, -0.3978,  0.0941,  0.3477],
#               [-0.2464, -1.6529, -0.0558,  0.6763],
#               [-0.5748, -0.1173, -0.7983, -0.5130]]])
```

**参数**：
- `*size`：整数序列，指定每个维度的大小

**相关方法**：
| 方法 | 说明 | 示例 |
|------|------|------|
| `torch.zeros()` | 全零张量 | `torch.zeros(2, 3)` → 全0 |
| `torch.ones()` | 全一张量 | `torch.ones(2, 3)` → 全1 |
| `torch.full()` | 指定值填充 | `torch.full((2,3), 5)` → 全5 |
| `torch.arange()` | 连续整数 | `torch.arange(10)` → 0到9 |
| `torch.linspace()` | 等差数列 | `torch.linspace(0, 1, 5)` → 0, 0.25, 0.5, 0.75, 1 |

---

### `torch.tensor()`

**作用**：从 Python 列表或数据创建张量

```python
# 从列表创建
x = torch.tensor([1, 2, 3])           # 1D tensor
x = torch.tensor([[1, 2], [3, 4]])    # 2D tensor

# 指定数据类型
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x = torch.tensor([1, 2, 3], dtype=torch.int64)

# 从 numpy 数组创建
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)
```

---

### `torch.manual_seed()`

**作用**：设置随机种子，确保结果可复现

```python
torch.manual_seed(42)  # 设置种子

# 每次设置相同种子后，随机数相同
torch.randn(3)  # tensor([0.3367, 0.1288, 0.2345])
torch.randn(3)  # tensor([0.2303, -1.1229, -0.1863])

torch.manual_seed(42)  # 重新设置
torch.randn(3)  # tensor([0.3367, 0.1288, 0.2345])  # 与第一次相同！
```

---

## 2. 数学运算方法

### `torch.matmul()` / `@`

**作用**：矩阵乘法

```python
# 2D 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # 结果形状 [3, 5]
# 或
C = A @ B

# 多维张量乘法 (batch 维度)
# [B, L, d_k] × [B, d_k, L] → [B, L, L]
Q = torch.randn(2, 3, 4)
K = torch.randn(2, 4, 3)
scores = torch.matmul(Q, K)  # 结果形状 [2, 3, 3]
```

**广播机制**：

```python
# 形状不同时，PyTorch 会自动广播
A = torch.randn(3, 4)    # [3, 4]
B = torch.randn(4)        # [4]
C = torch.matmul(A, B)   # [3] → A的每行 × B
```

**运算符对照表**：

| 运算符 | 方法 | 说明 |
|--------|------|------|
| `+` | `torch.add()` | 加法 |
| `-` | `torch.sub()` | 减法 |
| `*` | `torch.mul()` | 逐元素乘法 |
| `/` | `torch.div()` | 逐元素除法 |
| `@` | `torch.matmul()` | 矩阵乘法 |
| `**` | `torch.pow()` | 幂运算 |

---

### `torch.sum()`

**作用**：求和

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 求所有元素的和
torch.sum(x)  # tensor(21)

# 按维度求和
torch.sum(x, dim=0)  # tensor([5, 7, 9])  列和
torch.sum(x, dim=1)  # tensor([6, 15])    行和

# 保持维度
torch.sum(x, dim=1, keepdim=True)  # tensor([[6], [15]])
```

---

### `torch.mean()`

**作用**：求平均值

```python
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

torch.mean(x)              # tensor(3.5)
torch.mean(x, dim=0)      # tensor([2.5, 3.5, 4.5])
torch.mean(x, dim=1)      # tensor([2., 5.])
```

---

### `torch.max() / torch.min()`

**作用**：求最大值/最小值

```python
x = torch.tensor([[1, 5, 3], [4, 2, 6]])

# 全局最值
torch.max(x)    # tensor(6)
torch.min(x)    # tensor(1)

# 按维度最值
torch.max(x, dim=0)   # 返回 (values, indices)
# values: tensor([4, 5, 6])
# indices: tensor([1, 0, 1])

# 只返回值
torch.max(x, dim=1)[0]  # tensor([5, 6])
```

---

## 3. 张量变换方法

### `.transpose()`

**作用**：交换两个维度

```python
x = torch.randn(2, 3, 4)  # [2, 3, 4]

# 交换维度 1 和 2
y = x.transpose(1, 2)  # [2, 4, 3]

# 连续 transpose
y = x.transpose(0, 1).transpose(1, 2)  # [4, 3, 2]
```

**等同于**：

```python
x = torch.randn(2, 3, 4)
y = x.permute(0, 2, 1)  # permute 可以一次交换多个维度
```

---

### `.view()`

**作用**：改变张量形状（共享内存）

```python
x = torch.randn(2, 6)  # [2, 6]

# 展平
y = x.view(-1)  # [12]  -1 自动计算

# reshape
y = x.view(3, 4)  # [3, 4]

# 保持 batch 维度
y = x.view(x.size(0), -1)  # [2, 6]
```

**重要**：`.view()` 返回的是原张量的**视图**，共享内存！

```python
x = torch.randn(2, 4)
y = x.view(4, 2)
y[0, 0] = 999
print(x[0, 0])  # tensor(999.)  x 也被修改了！
```

---

### `.reshape()`

**作用**：改变形状（可能复制数据）

```python
x = torch.randn(2, 4)
y = x.reshape(4, 2)

# 与 view 的区别：
# - view 需要连续内存
# - reshape 会自动复制数据保证成功
```

---

### `.contiguous()`

**作用**：将张量转为连续内存存储

```python
x = torch.randn(2, 3, 4)

# transpose 后内存不连续
y = x.transpose(1, 2)  # [2, 4, 3]

# 需要 contiguous 才能使用 view
z = y.contiguous().view(2, 12)
```

---

### `.unsqueeze() / .squeeze()`

**作用**：添加/删除维度为1的维度

```python
x = torch.randn(3, 4)  # [3, 4]

# 添加维度
y = x.unsqueeze(0)  # [1, 3, 4]
y = x.unsqueeze(1)  # [3, 1, 4]
y = x.unsqueeze(-1) # [3, 4, 1]

# 删除维度
y = torch.randn(3, 1, 4)
z = y.squeeze()    # [3, 4]  删除所有为1的维度
z = y.squeeze(1)   # [3, 4]  只删除第1维（如果为1）
```

---

### `torch.cat()`

**作用**：沿指定维度拼接张量

```python
A = torch.randn(2, 3)
B = torch.randn(2, 3)

# 沿 dim=0 拼接
C = torch.cat([A, B], dim=0)  # [4, 3]

# 沿 dim=1 拼接
C = torch.cat([A, B], dim=1)  # [2, 6]
```

---

### `torch.stack()`

**作用**：沿新维度堆叠张量

```python
A = torch.randn(3, 4)
B = torch.randn(3, 4)

# 堆叠成新维度
C = torch.stack([A, B], dim=0)  # [2, 3, 4]
C = torch.stack([A, B], dim=1)  # [3, 2, 4]

# 与 cat 的区别
# cat: [3,4] + [3,4] = [6,4] (不增加维度)
# stack: [3,4] + [3,4] = [2,3,4] (增加新维度)
```

---

## 4. 神经网络层

### `nn.Linear()`

**作用**：线性变换（全连接层）

```python
linear = nn.Linear(in_features=4, out_features=8)

x = torch.randn(2, 3, 4)  # [batch, seq_len, in_features]
y = linear(x)              # [batch, seq_len, out_features]

# 内部操作: y = x × W^T + b
# W: [8, 4], b: [8]
```

**参数**：
- `in_features`：输入维度
- `out_features`：输出维度
- `bias`：是否使用偏置（默认 True）

---

### `nn.Sequential()`

**作用**：容器，按顺序执行多个层

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

x = torch.randn(2, 10)
y = model(x)  # [2, 10]
```

---

### `nn.ReLU() / nn.Sigmoid() / nn.Tanh()`

**作用**：激活函数

```python
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

x = torch.randn(2, 4)

y1 = relu(x)    # ReLU: max(0, x)
y2 = sigmoid(x) # Sigmoid: 1/(1+e^(-x))
y3 = tanh(x)   # Tanh: (e^x - e^(-x))/(e^x + e^(-x))
```

---

### `nn.Dropout()`

**作用**：随机丢弃神经元（防止过拟合）

```python
dropout = nn.Dropout(p=0.5)  # p 是丢弃概率

x = torch.randn(2, 10)
y = dropout(x)  # 随机将 50% 的元素置为 0
```

---

### `nn.LayerNorm()`

**作用**：层归一化

```python
layer_norm = nn.LayerNorm(normalized_shape=4)

x = torch.randn(2, 3, 4)  # [batch, seq_len, features]
y = layer_norm(x)         # 输出形状相同

# 对最后一个维度做归一化
# y = (x - mean) / std * gamma + beta
```

---

### `nn.Parameter()`

**作用**：将张量标记为可学习参数

```python
# 方式1：直接使用 tensor（需要手动注册）
self.weight = nn.Parameter(torch.randn(4, 4))

# 方式2：在 Module 的 __init__ 中定义
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(4))  # 可学习
        self.beta = nn.Parameter(torch.zeros(4))   # 可学习
```

---

## 5. 其他常用方法

### `F.softmax()`

**作用**：Softmax 归一化

```python
import torch.nn.functional as F

x = torch.tensor([2.0, 1.0, 0.1])

# 对最后一个维度 softmax
y = F.softmax(x, dim=-1)
# tensor([0.6590, 0.2424, 0.0986])
# 验证: 0.659 + 0.242 + 0.099 = 1.0
```

---

### `F.one_hot()`

**作用**：转换为 one-hot 编码

```python
classes = torch.tensor([0, 2, 1, 0])

# 转换为 one-hot
onehot = F.one_hot(classes, num_classes=3)
# tensor([[1, 0, 0],  # 0 → [1,0,0]
#         [0, 0, 1],  # 2 → [0,0,1]
#         [0, 1, 0],  # 1 → [0,1,0]
#         [1, 0, 0]]) # 0 → [1,0,0]
```

---

### `.detach()`

**作用**：从计算图分离（切断梯度）

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()  # z 不需要梯度

# y.requires_grad = True
# z.requires_grad = False
```

**用途**：从计算图中分离张量，用于不需要反向传播的场景。

---

### `.item()`

**作用**：将单元素张量转为 Python 标量

```python
x = torch.tensor(3.14)
print(x.item())  # 3.140000104904175

# 等价于
print(x.numpy().item())  # 3.14
```

---

### `.numpy()`

**作用**：将张量转为 NumPy 数组

```python
x = torch.randn(3, 4)
arr = x.numpy()  # numpy.ndarray

# 注意：共享内存！
arr[0, 0] = 999
print(x[0, 0])  # tensor(999.)  原张量也被修改！

# 避免共享内存
arr = x.detach().numpy()
```

---

### `.to()`

**作用**：移动张量到指定设备

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(3, 4)
x = x.to(device)  # 移动到 GPU

# 或者在创建时指定
x = torch.randn(3, 4, device=device)
```

---

## 练习中用到的方法速查表

| 文件 | 使用的方法 | 说明 |
|------|-----------|------|
| 1.1 | `torch.randn`, `torch.matmul`, `F.softmax`, `.shape`, `.sum`, `.transpose` | 基础运算 |
| 1.2 | `nn.Linear`, `nn.Sequential`, `F.one_hot`, `.view`, `.contiguous` | 多头注意力 |
| 1.3 | `nn.LayerNorm`, `nn.ReLU`, `nn.Dropout`, `.mean`, `.std` | BERT 层 |

---

## 常见错误及解决方案

### 1. 维度不匹配

```python
# 错误
Q = torch.randn(2, 3, 4)
K = torch.randn(2, 4)  # 缺少一个维度！

# 正确
K = torch.randn(2, 3, 4)  # 维度一致
```

### 2. view 需要连续内存

```python
# 错误
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2).view(2, 12)  # RuntimeError!

# 正确
y = x.transpose(1, 2).contiguous().view(2, 12)
```

### 3. requires_grad 问题

```python
# 错误：标量张量无法转 NumPy
x = torch.tensor(3.0)
x.numpy()  # RuntimeError! 需要 requires_grad=False

# 正确
x = torch.tensor(3.0, requires_grad=False)
arr = x.numpy()
```
