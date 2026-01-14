import re
import collections

# 1. BBPE 核心：字节到 Unicode 的映射 (标准 GPT-2 实现)
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()

# 2. 初始词频统计 (模拟 GPT-2 预分词逻辑)
def get_bbpe_vocab(text):
    # GPT-2 的核心正则简化版：匹配单词(可能带前导空格)、数字、标点或连续空格
    # 注意：' ?' 表示可选的空格前缀
    gpt2_pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")
    
    # 找出所有预分词块
    raw_chunks = re.findall(gpt2_pat, text)
    
    vocab = collections.Counter()
    for chunk in raw_chunks:
        # 将每个块转为字节，再映射为 Unicode 字符序列
        tokens = [byte_encoder[b] for b in chunk.encode('utf-8')]
        # 以空格分隔每个符号，用于 BPE 统计
        char_sequence = ' '.join(tokens)
        vocab[char_sequence] += 1
    return vocab

# 3. 统计相邻字符对频率 (保持不变)
def get_stats(vocab):
    pairs = collections.Counter()
    for word, frequency in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] += frequency
    return pairs

# 4. 执行合并逻辑 (保持不变)
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
# 语料库
text = "Hi " * 3 + "你好 " * 4 + "Hight " * 4
vocab = get_bbpe_vocab(text)

print("【初始 BBPE 词表 (已应用 Byte-mapping，无 </w>)】:")
# 这里的 'Ġ' 通常是空格 0x20 映射后的字符
for word, freq in vocab.items():
    print(f"'{word}': {freq}")
print("-" * 50)

# 执行合并
num_merges = 8
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = pairs.most_common(1)[0][0]
    vocab = merge_pairs(best_pair, vocab)
    
    # 将映射后的字符还原为可读显示
    readable_pair = "".join(best_pair)
    print(f"第 {i+1} 次合并: {best_pair} -> {readable_pair}")

print("-" * 50)
print("【合并后的结果】:")
for word, freq in vocab.items():
    print(f"'{word}': {freq}")
