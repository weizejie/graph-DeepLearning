本skill是有关于模型的具体实施的

主模型基于routenet-fermi-main下辖模型  但我们使用pytorch做为构筑

version-1.0.0

datagenerate
生成以flow为单位的单元batch

embedding：
flow features 同 routenet的处理模式
queue 以及 link也同routenet处理  使用routenet相同的特征
而后embedding投影至32维向量（同routenet）

tokenizer：
flow以 flow features的embedding后的token作为一个batch的句首

每个batch中，接下flow features token句首后的token 为一个hop token ， hop token 是二元的，一个组分是queue， 另一个组分是link 分别以其embedding的特征作为hop token的向量。

总结为每个batch以流为单位，会以flow feature本身作为句首token，跳作为后续的token（你也可以理解为一个流就是一个attention的滑动窗口）

embedding2：
除了原始的feature embedding， 接下来还会加入topology embedding
 1.相对位置embedding： 采用rope位置编码
 2.邻域特征embedding:  采用局部gnn，对于token的邻域信息进行整合处理，

encoder：
1.ffn层所有的线性变化里取消偏置参数
  非线性变化采用geglu
  拓展的维度是32*2.66维取整

预测的值，训练任务预测的值同routenet ，即预测每一跳的 rf
然后用logmse作为 loss，进行反向传播。