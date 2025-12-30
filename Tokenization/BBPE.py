import re,collections
#%%
import re
import collections

# 1. BBPE 核心：字节到 Unicode 的映射 (你的代码)
def bytes_to_unicode():
    # 挑选出“天生安全”的可打印字符区间
    #ord：字符 -> 数字; chr:数字 -> 字符
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    # 对于不在安全区（如空格 32）的字节，映射到 256 之后的新编码上
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 初始化映射表
byte_encoder = bytes_to_unicode()

# 2. 初始词频统计 (字节级)
def get_bbpe_vocab(text):
    vocab = collections.Counter()
    for word in text.strip().split():
        # 核心：先转 UTF-8 字节，再查表映射为可见符号
        tokens = [byte_encoder[b] for b in word.encode('utf-8')]
        char_sequence = ' '.join(tokens) + ' </w>'
        vocab[char_sequence] += 1
    return vocab

# 3. 统计相邻字符对频率
def get_stats(vocab):
    pairs = collections.Counter()
    for word, frequency in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] += frequency
    return pairs

# 4. 执行合并逻辑
def merge_pairs(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    replacement = ''.join(pair)
    for word in v_in:
        w_out = p.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out

# --- 模拟执行 ---
# 语料库：包含英文重叠和中文
text = "Hi " * 3 + "你好 " * 4 + "Hight " * 4
vocab = get_bbpe_vocab(text)

print("【初始 BBPE 词表】:")
print(vocab)
print("-" * 50)

# 执行 10 次合并
num_merges = 5
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = pairs.most_common(1)[0][0]
    vocab = merge_pairs(best_pair, vocab)
    print(f"第 {i+1} 次合并: {best_pair} -> {''.join(best_pair)}")

print("-" * 50)
print("【合并 10 次后的结果】:")
print(vocab)