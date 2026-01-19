from torchtext.data.utils import get_tokenizer

#获取一个分词器
'''
tokenizer – the name of tokenizer function
language
'''
tokenizer = get_tokenizer("spacy",language="en_core_web_sm")
print(tokenizer("Hello world!"))

#['Hello', 'world', '!']