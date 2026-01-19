from torchtext.data.utils import get_tokenizer



'''
为什么要使用yield
如果你这里用 return 返回一个包含所有分词结果的大列表，会发生什么？

内存崩溃 (OOM)：如果数据集有 100 万行，return 会强制要求程序一次性把这 100万行 
分词后的结果全部存入内存。

yield 的优势（生成器）：它像是一台“按需取货”的机器。
yield self.tokenizers 现场分词，给出一个结果，然后暂停。
等到需要下一个句子时，再继续。
内存始终只占用当前处理的一行数据。
在代码中作为参数传给词表构建函数
'''

def _yield_tokens(data, lang):
    # data_iter: 传入的原始数据集（比如 Hugging Face 的数据集对象）
    tokenizers = {
        'en': get_tokenizer('spacy', language='en_core_web_sm'),
        'de': get_tokenizer('spacy', language='de_core_news_sm'),
    }
    for data_sample in data:
        # data_sample: 数据集中的某一行，例如 {'en': 'Hello', 'de': 'Hallo'}

        #放回的是一个迭代对象
        yield tokenizers[lang](data_sample[lang])


# 模拟数据集 (data_iter)
data = [
    {"en": "I love AI.", "de": "Ich liebe KI."},
    {"en": "The cat sits.", "de": "Die Katze sitzt."},
    {"en": "Coding is fun.", "de": "Codieren macht Spaß."}
]

token_generator = _yield_tokens(data, "en")
for tokens in token_generator:
    print(tokens)
print("*"*30)
token_generator2 = _yield_tokens(data, "de")
for tokens in token_generator2:
    print(tokens)