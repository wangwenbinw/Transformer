'''
BPE主要有两个函数，一个统计Vocabulary,另一个合并字符串
'''
import re,collections
#%%
''' 统计词频'''
def get_vocab(text):
    """
    将原始文本转换为 BPE 初始词频字典，并添加终止符 </w>
    """
    vocab = collections.Counter()
    for word in text.strip().split():
        char_sequence = ' '.join(list(word)) + ' </w>'
        vocab[char_sequence] += 1
    return vocab

test_text = (
    "hug " * 5 +
    "pug " * 10 +
    "pun " * 12 +
    "bun " * 4
)

# 执行函数
test_vocab = get_vocab(test_text)
print(test_vocab)
# Counter({'p u n </w>': 12, 'p u g </w>': 10, 'h u g </w>': 5, 'b u n </w>': 4})
print('-'*50)
#%%
'''统计相邻字符对频率'''
def get_stats(vocab):
    pairs = collections.Counter()
    for word,frequency in vocab.items():
        # 将 'h u g </w>' 拆分为 ['h', 'u', 'g', '</w>']
        symbols = word.split()
        # 遍历单词内部的所有相邻组合
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] += frequency
    return pairs
pairs_stats = get_stats(test_vocab)
print(pairs_stats)
print('-'*50)
#%%
'''执行合并'''
def merge_pairs(pair,v_in):
    """
        在词频字典中将指定的字符对（pair）合并为一个新的子词
    """
    v_out = {}
    # 将元组 ('p', 'u') 转换为正则匹配模式 'p u'
    # re.escape 用于处理特殊字符（如终止符中的 /,确保被正确识别）
    bigram = re.escape(' '.join(pair))
    # ?<!:前面不能是，?!:后面不能是, \S 非空白字符
    #?<!\S = 前面不是非空白字符（即前面是空白或开头）
    #(?!\S) = 后面不是非空白字符（即后面是空白或结尾)
    p = re.compile(r'(?<!\S)'+bigram+r'(?!\S)')
    # 合并后的新字符连在一起，不留空格
    replacement = ''.join(pair)
    for word in v_in:
        w_out = p.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out


#%%
'''查阅算法执行10遍之后的结果'''
num_merges = 10
test_text = (
    "hug " * 5 +
    "pug " * 10 +
    "pun " * 12 +
    "bun " * 4
)
#获得词频表
vocab = get_vocab(test_text)
for i in range(num_merges):
    #获得相邻词频率
    pairs = get_stats(vocab)
    if not pairs:
        break
    #best_pair = max(pairs,key=pairs.get)
    best_pair = pairs.most_common(1)[0][0]
    vocab = merge_pairs(best_pair,vocab)
print(vocab)
#{'hug</w>': 5, 'pug</w>': 10, 'pun</w>': 12, 'bun</w>': 4}




