import torch
import torch.nn as nn
import torch.optim as optim

#%%准备模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        # Embedding层：将词索引映射为词向量,这里不是乘法计算，
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出层：从词向量预测目标词，线性层这里是乘法计算
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        """
        context_words: (batch_size, context_size=2*window_radius) 上下文词的索引
        """
        # 获取上下文词的embedding 
        embeds = self.embeddings(context_words)  # (batch_size, context_size, embedding_dim)

        # 对上下文词向量求平均,因为CBOW 的核心是将上下文的所有词向量压缩成一个向量。这里使用的是求平均 (Mean)。
        mean_embeds = torch.mean(embeds, dim=1)  # (batch_size, embedding_dim)

        # 预测目标词，
        out = self.linear(mean_embeds)  # (batch_size, vocab_size)

        return out


#%%准备训练数据
def create_cbow_dataset(text,window_size=2):
    """
        text: 分词后的文本列表
        window_size: 上下文窗口大小
    """
    data = []
    for i in range(window_size,len(text)-window_size):
        context = text[i-window_size:i] + text[i+1:i+1+window_size]
        target = text[i]
        data.append((context, target))
    return data


#%%
text = "I love natural language processing and deep learning".split()
vocab = list(set(text))
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)
print(word_to_idx)
print([word_to_idx[w] for w in text])
print('*'*20)
#%% 创建数据集
dataset = create_cbow_dataset([word_to_idx[w] for w in text], window_size=2)

#%%训练
model = CBOW(vocab_size,embedding_dim=50)
optimizer = optim.SGD(model.parameters(),lr=0.01)
loss_function =nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    for context, target in dataset:
        #准备数据
        comtext_tensor = torch.tensor([context],dtype=torch.long)
        target_tensor = torch.tensor([target],dtype=torch.long)
        #前向传播
        model.zero_grad()
        output = model(comtext_tensor)
        #计算损失
        loss = loss_function(output, target_tensor)
        #反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {total_loss:.4f}')

#%% 获取训练好的词向量
word_embeddings = model.embeddings.weight.data
print(f"\n词向量矩阵形状: {word_embeddings.shape}")
