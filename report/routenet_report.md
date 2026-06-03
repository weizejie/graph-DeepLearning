#    RouteNet-Fermi 模型输入数据格式与处理流程深度分析报告

生成日期：2026-05-27

## 1. 项目概述

RouteNet-Fermi 是一个基于 TensorFlow 的图神经网络（GNN）模型，专门用于预测
计算机网络的性能指标（如时延 Delay、抖动 Jitter、数据包丢失 Packet Loss）。
该项目由加泰罗尼亚收术理工大学开发，是 RouteNet 系列的最新版本。

与经典 GNN 不同，RouteNet-Fermi 处理的是异构图（Hypergraph），其中包含三种
不同类型的节点：Path（路径）、Link（链路）、Queue（队列），并通过自定义的
消息传递机制在节点间交换信息。

## 2. 原始数据存储格式

## 2.1 目录结构
--------------------------------------------------------------------------------
注册内容：
    - simulationResults.txt  仿真性能结果（全局 + 流量）
    - traffic.txt            流量矩阵参数
    - stability.txt          仿真稳定性状态
    - input_files.txt         图文件和路由文件名
    - flowSimulationResults.txt 流级仿真结果（可选）
    - linkUsage.txt           链路利用率统计（可选）

数据集分类：
#    fat128/              Fat-Tree 128 节点拓扑（合成数据）
#    real_traces/         真实网络轨迹数据
#        train/geant/     GEANT 欧洲学术网络训练集
#        test/abilene/    Abilene 美国学术网络
#        test/geant/      GEANT 测试集
#        test/germany50/   德国网络50节点
#        test/nobelp/      Nobel 网络

## 2.2 网络拓扑文件（GML 格式）
--------------------------------------------------------------------------------
GML（Graph Modeling Language）格式存储网络结构。包含节点和边属性。

节点属性：
#    id                  节点唯一标识符（int）
#    schedulingPolicy     队列调度策略（string）：WFQ/加权公平队列、SP/严格优先级、DRR/差分轮询、FIFO/先进先出
#    schedulingWeights    调度权重（WFQ/DRR 使用）
#    levelsQoS           QoS 队列数量（通常 1~3）
#    bufferSizes          每个 QoS 队列的缓冲区大小（bytes）

边属性：
#    source              源节点 ID
#    target              目的节点 ID
#    bandwidth           链路带宽（bps）
#    port                端口号
#    weight              路由权重

## 2.3 路由文件（TXT 格式）
--------------------------------------------------------------------------------
描述每个源节点到每个目的节点的输出端口号矩阵。值是从源节点
到目的节点应该使用的输出端口号。-1表示不可达。DatanetAPI
将端口号转换为下一跳节点 ID，最终生成路径列表。

## 2.4 仿真结果文件（simulationResults.txt）
--------------------------------------------------------------------------------
格式：全局统计|通节点对统计;...
解析后得到：
    - 全局统计：global_packets, global_losses, global_delay
    - 通节点对统计：PktsDrop, AvgDelay, AvgLnDelay, p10-p90, Jitter
    - 流级结果：每个 OD 对中各条流的性能指标

## 2.5 流量参数文件（traffic.txt）
--------------------------------------------------------------------------------
格式：maxAvgLambda|通OD对流量;...

时间分布参数（TimeDist，共 8 种）：
#    0   EXPONENTIAL_T        EqLambda, AvgPktsLambda, ExpMaxFactor
#    1   DETERMINISTIC_T     EqLambda, AvgPktsLambda
#    2   UNIFORM_T           EqLambda, MinPktLambda, MaxPktLambda
#    3   NORMAL_T            EqLambda, AvgPktsLambda, StdDev
#    4   ONOFF_T             EqLambda, PktsLambdaOn, AvgTOff, AvgTOn, ExpMaxFactor
#    5   PPBP_T              EqLambda, BurstGenLambda, Bitrate, Pareto参数
#    6   TRACE_T             EqLambda（真实轨迹）
#    7   EXTERNAL_PY_T      EqLambda, 自定义分布参数

包大小分布参数（SizeDist，共 5 种）：
#    0   DETERMINISTIC_S     AvgPktSize
#    1   UNIFORM_S           AvgPktSize, MinSize, MaxSize
#    2   BINOMIAL_S          AvgPktSize, PktSize1, PktSize2
#    3   GENERIC_S           AvgPktSize, NumCandidates, Size_i, Prob_i
#    4   TRACE_S             AvgPktSize（从流量计算）

## 3. 数据处理流程

## 3.1 流程总览
--------------------------------------------------------------------------------
Step 1: 遍历 .tar.gz 压缩包，逐行解析仿真输出
Step 2: 读取 GML 拓扑文件，构建 NetworkX 有向图
Step 3: 解析路由矩阵文件，将端口号转换为节点路径列表
Step 4: 解析流量和性能矩阵，填充 Sample 对象
Step 5: 调用 network_to_hypergraph() 构建异构图
Step 6: 调用 hypergraph_to_input_data() 转换为 RaggedTensor 特征字典
Step 7: TensorFlow 模型训练/推理

## 3.2 DatanetAPI 解析层
--------------------------------------------------------------------------------
DatanetAPI 是数据集的核心解析器，封装在 datanetAPI.py 中：

#    tool = DatanetAPI(data_dir, shuffle=shuffle)
#    for sample in tool:
#        G = sample.get_topology_object()      -> NetworkX 有向图
#        T = sample.get_traffic_matrix()       -> NxN 流量矩阵
#        R = sample.get_routing_matrix()       -> NxN 路由矩阵
#        P = sample.get_performance_matrix()   -> NxN 性能矩阵

核心职责：遍历 .tar.gz、读取GML拓扑、解析路由、解析流量性能、支持过滤。

## 3.3 网络到超图的转换（network_to_hypergraph）
--------------------------------------------------------------------------------
将原始网络数据转换为包含三类节点的有向异构图。

三类节点：

#    1. 路径节点 (Path) p_{src}_{dst}_{flow_id}
#       对应每一对源-目的节点间的每一条流。
#       属性：traffic（平均带宽）、packets（包率）、model（时间分布类型）、eq_lambda等

#    2. 链路节点 (Link) l_{node1}_{node2}
#       对应拓扑中的每一条有向边。属性：capacity（带宽）、policy（调度策略）

#    3. 队列节点 (Queue) q_{node1}_{node2}_{q_id}
#       对应每个节点上每个输出方向的每个 QoS 队列。
#       属性：queue_size（队列大小）、priority（优先级）、weight（权重）

## 4. TensorFlow 模型输入规格

## 4.1 特征维度总览
--------------------------------------------------------------------------------
共 21 个输入特征

【路径特征 11个原始 + 7个 one-hot = 17维】
#    traffic             [num_paths, 1]    路径平均带宽（bps）
#    packets             [num_paths, 1]    路径包生成率（pkt/s）
#    model[one-hot 7]   [num_paths, 7]    时间分布类型（one-hot）
#    eq_lambda          [num_paths, 1]    等效到达率
#    avg_pkts_lambda    [num_paths, 1]    平均包到达率
#    exp_max_factor     [num_paths, 1]    指数最大因子
#    pkts_lambda_on     [num_paths, 1]    ON期间包到达率
#    avg_t_off          [num_paths, 1]    OFF 平均持续时间
#    avg_t_on           [num_paths, 1]    ON 平均持续时间
#    ar_a               [num_paths, 1]    AR(1) 自相关系数
#    sigma              [num_paths, 1]    AR(1) 标准差

【链路特征 1个原始 + 4个 one-hot = 5维】
#    capacity            [num_links, 1]    链路带宽（bps）
#    policy[one-hot 4] [num_links, 4]    调度策略（WFQ/SP/DRR/FIFO）

【队列特征 1个原始 + 3个 one-hot + 1 = 5维】
#    queue_size          [num_queues, 1]   队列缓冲区大小（bytes）
#    priority[one-hot 3][num_queues, 3]   队列优先级（one-hot）
#    weight              [num_queues, 1]   队列权重（WFQ/DRR 使用）

【拓扑关系 5个 RaggedTensor】
#    link_to_path        Ragged [num_links]    链路经过的路径列表
#    queue_to_path       Ragged [num_queues]   队列服务的路径列表
#    path_to_queue       Ragged [num_paths, 2] 路径经过的队列（含位置）
#    queue_to_link       Ragged [num_queues]   队列所属的链路
#    path_to_link        Ragged [num_paths, 2] 路径经过的链路（含位置）

【辅助特征】
#    length              [num_paths]         路径跳数（用于 ragged 恢复）

【标签】
#    delay               [num_paths]         路径平均时延（秒）——预测目标

## 4.2 RaggedTensor（不规则张量）
--------------------------------------------------------------------------------
不同路径经过的链路/队列数量不同，模型使用 RaggedTensor 处理变长序列。
避免固定 padding 带来的内存浪费。指定长度信息，GRU 支持变长输入。

## 4.3 拓扑关系映射表
--------------------------------------------------------------------------------
#    link_to_path     每条链路被哪些路径经过        RaggedTensor[num_links]
#    queue_to_path    每个队列服务哪些路径          RaggedTensor[num_queues]
#    path_to_queue    每条路径经过哪些队列（含位置） RaggedTensor[num_paths, 2]
#    queue_to_link    每个队列属于哪条链路          RaggedTensor[num_queues]
#    path_to_link     每条路径经过哪些链路（含位置） RaggedTensor[num_paths, 2]

## 5. 模型架构中的数据流动

## 5.1 Embedding 层
--------------------------------------------------------------------------------
每种节点类型通过两层 Dense+ReLU 将特征映射到 32维度隐藏状态：

#    路径 Embedding：17维 -> 32维
#    path_state = self.path_embedding(tf.concat([
#        (traffic - mean) / std,           Z-Score归一化后的流量、包率、one-hot编码、时间分布参数
#    ], axis=1))

#    链路 Embedding：5维 -> 32维
#    link_state = self.link_embedding(tf.concat([load, policy], axis=1))
#    其中 load = SUM(路径流量) / 链路容量

#    队列 Embedding：5维 -> 32维
#    queue_state = self.queue_embedding(tf.concat([
#        (queue_size - mean) / std, tf.one_hot(priority, 3), weight
#    ], axis=1))

## 5.2 8轮消息传递
--------------------------------------------------------------------------------
RouteNet-Fermi 执行 8轮迭代的消息传递，每轮包含三个步骤：

#    for iteration in range(8):

#      步骤1: Link & Queue -> Path
#        queue_state --gather--> queue_gather
#        link_state  --gather--> link_gather
        --> GRU Cell --> path_state 更新

#      步骤2: Path -> Queue
#        path_state_sequence --gather_nd--> path_gather --> sum --> GRU --> queue_state 更新

#      步骤3: Queue -> Link
#        queue_state --gather--> queue_gather --> RNN --> link_state 更新

## 5.3 Readout（输出层）
--------------------------------------------------------------------------------
经过 8轮消息传递后，MLP 预测时延：
#    input_tensor = path_state_sequence[:, 1:].to_tensor()   [P, 8, 32]
#    occupancy_gather = self.readout_path(input_tensor)       [P, 8, 1]
#    occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)
#    queue_delay = sum(occupancy / capacity)
#    trans_delay = pkt_size * sum(1 / capacity)
#    return queue_delay + trans_delay

## 6. 训练流程

#    ds_train = input_fn(TRAIN_PATH, shuffle=True)
#    ds_train = ds_train.prefetch(tf.data.AUTOTUNE).repeat()
#    捕失函数：MAPE（Mean Absolute Percentage Error）
#    优化器：Adam（学习率0.001）
#    model.fit(ds_train, epochs=50, steps_per_epoch=2000, validation_data=ds_validation)

数据集：
#    fat128/train              Fat-Tree 128节点   合成流量
#    real_traces/train/geant   22节点          真实网络轨迹
#    real_traces/test/*        各测试集          真实网络轨迹

Z-Score 归一化参数：
#    traffic  均值:1385.41 标准差:859.81
#    packets  均值:1.40    标准差:0.89
#    capacity 均值:27611.09 标准差:20090.62
#    queue_size 均值:30259.11 标准差:21410.10

## 7. 与标准 GNN 的关键区别

#    维度             标准 GNN              RouteNet-Fermi
#    节点类型       单类型（同构图）      三类型（Path/Link/Queue）
#    边关系         1:1（简单边）           N:1, 1:N, N:N（复杂超图）
#    序列长度       固定                  可变（RaggedTensor）
#    消息传递       聚合邻居状态         gather+sum+RNN（顺序感知）
#    路由感知       无                   路径节点编码路由信息
#    输入数据       节点/边特征           拓扑+流量+路由+队列多维特征
#    物理意义       通用图表学习         排队论的网络排队模型

## 8. 关键设计决策分析

## 8.1 为什么选择超图结构？
--------------------------------------------------------------------------------
计算机网络天然具有多层次结构：路径层（端到端流）、链路层（带宽限制）、
队列层（排队延负）。三层结构使模型能分别学习流量特性、择堵竞争和调度策略。

## 8.2 为什么使用 RaggedTensor？
--------------------------------------------------------------------------------
网络路径长度不一是 GNN 领域的经典难题。不用 padding 避免引入虚假信息，
RaggedTensor 保留真实长度信息，GRU 支持变长输入。

## 8.3 为什么用 8轮迭代？
--------------------------------------------------------------------------------
消息传递的轮数需要平衡：足够多（信息从链路/队列传播到路径需要多跳）
与不过多（过深导致过度平滑）。(8轮是经过验证的折中值）

## 8.4 为什么 Z-Score 而非 Min-Max？
--------------------------------------------------------------------------------
Z-Score 保留数据的相对分布特性，适用于有物理意义的流量参数（带宽、时延等），
不会因异常值而过度压缩有效范围。

报告生成完毕
生成时间：2026-05-27