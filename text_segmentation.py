import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn as nn


print("tiktoken version:", version("tiktoken"))

file_path = "the-verdict.txt"
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


tokenizer = tiktoken.get_encoding("gpt2")

class GPTDatasetV1(Dataset):
    #参数
    #txt: 原始文本数据
    #tokenizer: 分词器，用于将文本转换为token ID序列
    #max_length: 每个训练样本的最大长度（上下文窗口大小）
    #stride: 滑动窗口的步长
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        #首先将整个文本txt通过分词器转换为token ID序列
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            #input_chunk: 从位置i开始，长度为max_length的token序列
            input_chunk = token_ids[i:i + max_length]
            #从位置i+1开始，长度为max_length的token序列（即输入序列向右偏移1位）
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids) 
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
#这个函数通常用于：
#准备 GPT 风格语言模型的训练数据
#处理长文本时使用滑动窗口创建训练样本
#批量加载数据到模型中训练
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    #使用 OpenAI 的 tiktoken 库获取 GPT-2 的 tokenizer，将文本转换为 token ID                         
    tokenizer = tiktoken.get_encoding("gpt2")
    #创建一个 GPTDatasetV1实例，这个数据集类应该实现了：
    #将文本 tokenize 成 token ID
    #使用滑动窗口方法创建训练样本
    #每个样本长度为 max_length，相邻样本重叠 max_length - stride 个 token
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    #使用 PyTorch 的 DataLoader 包装数据集，提供批量加载、打乱、多进程加载等功能
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                drop_last=drop_last, num_workers=num_workers)
    return dataloader

#max_length = 4: 设置上下文窗口大小为4个token
max_length = 4
#创建数据加载器，从文本中生成训练样本
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)
data_iter = iter(dataloader)
#获取一个批次的数据：inputs是输入序列，targets是目标序列（输入向右偏移1位）
inputs, targets = next(data_iter)

#词嵌入层
#vocab_size = 50257: GPT-2的词表大小
vocab_size = 50257
#output_dim = 256: 嵌入向量的维度
output_dim = 256
#创建词嵌入层，将token ID映射为256维向量
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#token_embeddings.shape: 输出形状为 (batch_size, sequence_length, embedding_dim)
token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec

class SelftAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # print("keys shape:", keys.shape)
        # print("queries shape:", queries.shape)
        # print("values shape:", values.shape)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        
        return context_vec

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        b, num_tokens, d_in = x.shape 
        keys = self.W_key(x) 
        queries = self.W_query(x) 
        values = self.W_value(x) 

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)   

        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2) 

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        print("context_vec shape:", context_vec.shape)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


        

inputs = torch.tensor([
    [0.43, 0.15, 0.89], #Your
    [0.55, 0.87, 0.66], #journey
    [0.57, 0.85, 0.64], #starts
    [0.22, 0.58, 0.33], #with
    [0.77, 0.25, 0.10], #one
    [0.05, 0.80, 0.55]  #step
])

batch = torch.stack([inputs, inputs], dim=0)

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)







