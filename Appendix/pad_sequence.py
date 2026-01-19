import torch
from torch.nn.utils.rnn import pad_sequence
'''
：将一组长度不一的张量（Tensors）对齐，
通过填充（Padding）把它们变成一个形状整齐的单一矩阵。
自动补长到最长的一句
'''

# 模拟三句不同长度的索引序列
s1 = torch.tensor([2, 5, 8, 3])       # 长度 4
s2 = torch.tensor([2, 10, 3])         # 长度 3
s3 = torch.tensor([2, 12, 15, 20, 3]) # 长度 5

seq_list = [s1, s2, s3]
print(seq_list)

# 执行填充，设定填充值为 1 (代表 <pad>)
'''
padding_value=指定用什么数字来填充。
batch_first=True:如果为 True：输出形状是 [Batch大小, 句子长度]。这是 Transformer 和大多数现代模型喜欢的格式。

'''
padded_batch = pad_sequence(seq_list, batch_first=True, padding_value=1)
print(padded_batch)
# 输出结果：
# tensor([[ 2,  5,  8,  3,  1],  <- 补了 1 个 1
#         [ 2, 10,  3,  1,  1],  <- 补了 2 个 1
#         [ 2, 12, 15, 20,  3]]) <- 最长的一句，不补