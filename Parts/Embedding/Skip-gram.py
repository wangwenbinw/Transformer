import torch
import torch.nn as nn
import torch.optim as optim


#%%
class SkipGram(nn.Module):
    def __init__(self,vocab_size,embedding_dim=50):
        super(SkipGram,self).__init__()
        # Embedding层：输入层到隐藏层
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        # 输出层
        self.linear = nn.Linear(embedding_dim,vocab_size)

    def forward(self,center_word):
        '''
        centre_word: 中心词的索引
        '''
        # 获取中心词的embedding
        embeds = self.embeddings(center_word)  # (batch_size, embedding_dim)

        # 预测上下文词
        out = self.linear(embeds)  # (batch_size, vocab_size)

        return out

#%% 创建skip-gram数据集
def create_skipgram_dataset(text,window_size=2):
    """
        text: 词索引列表
        window_size: 上下文窗口大小
        返回: [(center_word, context_word), ...]
    """
    data = []
    for i in range(len(text)):
        start = max(0,i-window_size)
        end = min(len(text),i+window_size)
        for j in range(start,end):
            if i!=j:
                center = text[i]
                context = text[j]
                data.append((center,context))
        return data

#%%
text = "I love natural language processing and deep learning with neural networks".split()
vocab = list(set(text))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocab)

#%%
text_indices = [word_to_idx[w] for w in text]
dataset = create_skipgram_dataset(text_indices, window_size=2)
print(f"词汇表大小: {vocab_size}")
print(f"训练样本数: {len(dataset)}")
#%%
embedding_dim = 50
model = SkipGram(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

# 训练
epochs = 100
for epoch in range(epochs):
    total_loss = 0

    for center, context in dataset:
        # 准备数据
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)

        # 前向传播
        model.zero_grad()
        output = model(center_tensor)

        # 计算损失
        loss = loss_function(output, context_tensor)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

# 获取训练好的词向量
word_embeddings = model.embeddings.weight.data
print(f"\n词向量矩阵形状: {word_embeddings.shape}")

#%%计算相似度
def cosine_similarity(word1, word2):
    idx1 = word_to_idx[word1]
    idx2 = word_to_idx[word2]
    vec1 = word_embeddings[idx1]
    vec2 = word_embeddings[idx2]
    return torch.cosine_similarity(vec1, vec2, dim=0).item()

# 测试
print("\n词相似度：")
print(f"'language' 和 'processing': {cosine_similarity('language', 'processing'):.4f}")
print(f"'deep' 和 'learning': {cosine_similarity('deep', 'learning'):.4f}")

