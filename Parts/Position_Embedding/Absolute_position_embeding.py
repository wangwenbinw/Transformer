import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000,dropout=0.1):
        '''
        参数
        :param d_model: 词向量维度 (embedding dimension)
        :param max_len: 预计算的最大序列长度 (通常设大一点，比如 5000
        :param dropout: 加完位置编码后的 dropout 概率
        '''
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        #1.初始化一个全0矩阵[max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        #2.生成位置索引 [0, 1, ..., max_len-1]
        # shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        #3.计算分母的div_term
        #公式: 1 / (10000 ^ (2i / d_model))
        #在对数空间计算 exp(log(...)) 可以提高数值稳定性
        # div_term shape: [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0))


        #4填充矩阵
        #偶数位sin(pos * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位: cos(pos * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #5. 增加一个维度，方便广播
        # shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 6. 注册为 buffer
        # 这一步非常关键：
        # - 它不是模型参数 (Parameter)，不需要梯度更新 (requires_grad=False)
        # - 但它需要随模型保存 (state_dict 里面要有它)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
                x: [batch_size, seq_len, d_model]
        """
        # 1. 根据当前输入的实际长度 seq_len，切取对应长度的位置编码
        # x.size(1) 就是 seq_len
        # self.pe 的形状是 [1, max_len, d_model]，切片后自动广播到 batch 维度
        x = x + self.pe[:, :x.size(1)]

        # 2. 加上 Dropout (标准 Transformer 操作)
        return self.dropout(x)



