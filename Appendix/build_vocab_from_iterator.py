from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np
class TranslationProcessor:
    def __init__(self, src_lang='en', trg_lang='de', max_len=256):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_len = max_len

        # 加载分词器,spacy智能分词
        self.tokenizers = {
            'en': get_tokenizer('spacy', language='en_core_web_sm'),
            'de': get_tokenizer('spacy', language='de_core_news_sm'),
        }

        self.special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.UNK_IDX, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.vocabs = {}

    def _yield_tokens(self, data, lang):
        for data_sample in data:
            yield self.tokenizers[lang](data_sample[lang])

    def build_vocabs(self, train_data):
        for lang in [self.src_lang, self.trg_lang]:
            vocab = build_vocab_from_iterator(
                #一个能够产生 Token 列表的迭代器
                self._yield_tokens(train_data, lang),
                #只有在整个数据集中出现次数 $\ge 2$ 的单词才有资格进入词表
                min_freq=1,
                #强行插入一些特殊的标记。这些词可能在原始文本中不存在，但模型训练必不可少（如填充符 <pad>）
                specials=self.special_symbols,
                #确保这些特殊符号的索引（Index）是从 0, 1, 2, 3... 开始的。这在编写代码时非常方便，因为你可以一眼看出 0 代表什么。
                special_first=True
            )
            # 为那些不在词表里的“陌生词”设置一个默认的安放处
            vocab.set_default_index(self.UNK_IDX)
            self.vocabs[lang] = vocab
        print("词汇表构建完成。")

# 模拟数据集 (data_iter)
data = [
    {"en": "I love AI.", "de": "Ich liebe KI."},
    {"en": "The cat sits.", "de": "Die Katze sitzt."},
    {"en": "Coding is fun.", "de": "Codieren macht Spaß."}
]

translation_processor = TranslationProcessor(src_lang='en', trg_lang='de')
translation_processor.build_vocabs(data)
# 单词列表 -> 索引列表
print(translation_processor.vocabs['en'].lookup_indices(['I','love']))
# 索引列表-> 单词列表
print(translation_processor.vocabs['en'].lookup_tokens([7,12]))

print(len(translation_processor.vocabs['en']))
print(translation_processor.vocabs['en'].lookup_tokens(np.arange(0,14)))