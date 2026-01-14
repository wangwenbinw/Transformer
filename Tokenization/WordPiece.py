import re,collections
#%%
''' 1. 统计词频 (WordPiece 风格) '''
def get_wp_vocab(text):
    """
    WordPiece 初始化：首字母保持原样，后续字母加上 ## 前缀
    """
    vocab = collections.Counter()
    for word in text.strip().split():
        chars = list(word)
        wp_sequence = [chars[0]] + ["##" + c for c in chars[1:]]
        print(wp_sequence)
        vocab[' '.join(wp_sequence)] += 1
    return vocab
text = 'I love you'
vocab = get_wp_vocab(text)
print(vocab)
#%%
''' 2. 统计 Token 频次与 Pair 频次 '''


def get_wp_stats(vocab):
    """
    WordPiece 需要同时知道：
    1. 单个 Token 的频次 (用于分母)
    2. 相邻对 Pair 的频次 (用于分子)
    """
    token_counts = collections.defaultdict(int)
    pair_counts = collections.defaultdict(int)

    for word, frequency in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)):
            token_counts[symbols[i]] += frequency
            if i < len(symbols) - 1:
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += frequency
    return token_counts, pair_counts

#%%
''' 3. 计算 WordPiece 得分 '''
def calculate_wp_scores(token_counts, pair_counts):
    """
    公式: Score = Count(AB) / (Count(A) * Count(B))
    """
    scores = {}
    for pair, pair_freq in pair_counts.items():
        token_a, token_b = pair
        # 计算得分
        score = pair_freq / (token_counts[token_a] * token_counts[token_b])
        scores[pair] = score
    return scores


# %%
''' 4. 执行合并 '''
def merge_wp_pairs(pair, v_in):
    v_out = {}
    # re.escape:字符安全转义
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    # 合并逻辑：如果是 A + ##B -> AB; 如果是 ##A + ##B -> ##AB
    a, b = pair
    replacement = a + b.replace("##", "")

    #能够匹配，执行合并
    #无法匹配，原样输出
    for word in v_in:
        w_out = p.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out


# %%
''' 5. 模拟训练过程 '''
num_merges = 5
test_text = (
        "hug " * 5 +
        "pug " * 10 +
        "pun " * 12 +
        "bun " * 4
)

# 初始词频
vocab = get_wp_vocab(test_text)
print("【初始词表】:", vocab)

for i in range(num_merges):
    # 统计频次
    token_counts, pair_counts = get_wp_stats(vocab)
    if not pair_counts:
        break

    # 计算似然得分并选出最高分
    scores = calculate_wp_scores(token_counts, pair_counts)
    best_pair = max(scores, key=scores.get)

    print(f"第 {i + 1} 次合并: {best_pair} | 得分: {scores[best_pair]:.4f}")

    # 执行合并
    vocab = merge_wp_pairs(best_pair, vocab)

print("-" * 50)
print("【最终结果】:")
print(vocab)