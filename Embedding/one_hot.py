
vocab = ['apple','banana','cherry']
#构造一个词到索引的映射
word_to_index = {word: idx for idx,word in enumerate(vocab)}
#定义one-hot函数
def one_hot_encoding(word,vocab,word_to_index):
    #创建一个与词汇表长度相等的全0向量
    encoding = [0]*len(vocab)
    #获取该词索引,如果没有这个键就返回-1
    idx = word_to_index.get(word,-1)
    if idx != -1:
        encoding[idx] = 1
    return encoding
#测试
print('OneHot编码')
for word in vocab:
    print(f'word: {one_hot_encoding(word,vocab,word_to_index)}')
