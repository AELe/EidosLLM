import re
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

file_path = "the-verdict.txt"
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])
#分割文本
# - re.split() ：Python的 re 模块中的分割函数，根据正则表达式模式分割字符串
# - 正则表达式模式 ： r'([,.:;?_!"()\']|--|\s)'

# - r'' ：原始字符串，避免转义字符的问题
# - () ：捕获分组，表示分割时保留分隔符
# - [,.:;?_!"()\'] ：匹配以下任意一个标点符号：
#   - , 逗号
#   - . 句号
#   - : 冒号
#   - ; 分号
#   - ? 问号
#   - _ 下划线
#   - ! 感叹号
#   - " 双引号
#   - () 括号
#   - ' 单引号
# - | ：或运算符
# - -- ：匹配双破折号（英文中常用的标点）
# - \s ：匹配任何空白字符（空格、制表符、换行符等）
# - 功能 ：将文本按照标点符号和空白字符进行分割，同时保留这些分隔符

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# - 列表推导式 ： [item.strip() for item in preprocessed if item.strip()]
# - item.strip() ：移除每个元素两端的空白字符
# - if item.strip() ：过滤掉空字符串或只包含空白字符的元素
# - 功能 ：清理分割结果，移除空白元素，只保留有内容的词和标点
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# 输出分割后的元素总数，即文本被分割成了多少个词/标点
# print(len(preprocessed))
# 输出前30个分割后的元素，用于查看分割效果
# print(preprocessed[:30])
# 构建词汇表的过程。首先将训练集中的全部文本分词成独立的词元；
# 然后将这些词元按字母顺序进行排列，并删除重复的词元；
# 接下来将唯一的词元聚合到一张词汇表中，该词汇表定义了每个唯一的词元到唯一的整数值的映射。

# 创建一个包含所有唯一词元的列表，并将它们按照字母顺序排列
# - sorted() ：对可迭代对象进行排序，返回一个新的有序列表
# - set() ：创建一个无重复元素的集合，用于删除重复的词元
# - 功能 ：将分割后的元素转换为一个有序的列表，同时删除重复的词元
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
# print(vocab_size)

#创建词汇表
# 创建一个字典，其中键是词元（token），值是整数索引
# enumerate() 函数遍历 all_words 列表，返回(索引, 元素)对
vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)
for i, item in enumerate(vocab.items()):
    # print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        # vocab - 词汇表字典，格式为 {词元: 索引}
        # 保存原始词汇表，用于将字符串映射到整数
        self.str_to_int = vocab
        # 创建反向映射字典，用于将整数映射回字符串
        self.int_to_str = {i : s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # 对每个 item 执行：
        # 1. 计算 item.strip() 得到清理后的结果
        # 2. 检查 item.strip() 是否为空
        # 3. 如果不为空，将清理后的结果加入新列表
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 执行顺序 是从右到左的
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        # 用空格连接列表中的所有词元
        ### 在正则表达式 r'\s+([,.?!"()\'])' 中：
        # - () 创建了一个 捕获组
        # - 这个捕获组匹配标点符号： , . ? ! " ( ) '
        # - 这是 第一个 （也是唯一一个）捕获组
        ### 2. r'\1' 的含义：
        # - \1 表示"第一个捕获组的内容"
        # - 在替换时，用捕获组匹配到的标点符号本身来替换
        ### 3. 实际效果：
        # 匹配到： " ," （空格+逗号）

        # - 捕获组匹配到： , （逗号）
        # - \1 就是： , （逗号）
        # - 替换结果：用 "," 替换 " ,"
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
# vocab 是 词汇表 （vocabulary 的缩写），它是一个字典，将每个唯一的词元（token）映射到一个唯一的整数索引
# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"
#        Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))    

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))
# print(text)
# 创建了一个 GPT-2 模型的 tokenizer 对象，该对象可以用于：

# - 编码（encode） ：将文本字符串转换为 token ID 列表
# - 解码（decode） ：将 token ID 列表转换回文本字符串
tokenizer = tiktoken.get_encoding("gpt2")
# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#      "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)
text = ("Akwirw ier")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)


