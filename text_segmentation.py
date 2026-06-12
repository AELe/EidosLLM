import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn as nn
import matplotlib.pyplot as plt


print("tiktoken version:", version("tiktoken"))

file_path = "the-verdict.txt"
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(raw_text)
total_tokens = len(tokenizer.encode(raw_text))



class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


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
        # print("keys shapprinte:", keys.shape)
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
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        print("d_in:", d_in, "d_out:", d_out)
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

        print("b:", b, "num_tokens:", num_tokens, "num_heads:", self.num_heads, "head_dim:", self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)     
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)      

        keys = keys.transpose(1, 2)  
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)  

        attn_scores = queries @ keys.transpose(2, 3) 
        print(keys.transpose(2, 3))
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] 
        attn_scores.masked_fill_(mask_bool, -torch.inf) 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)  
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) 
        context_vec = self.out_proj(context_vec) 
        return context_vec

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) 
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

 
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seg_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(torch.arange(seg_len, device=in_idx.device))
     
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  
        var = x.var(dim=-1, keepdim=True, unbiased=False) 
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)

class ExampleDeepNeuralNetWork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[0], layer_sizes[1], GELU()),
                                     nn.Linear(layer_sizes[1], layer_sizes[2], GELU()),
                                     nn.Linear(layer_sizes[2], layer_sizes[3], GELU()),
                                     nn.Linear(layer_sizes[3], layer_sizes[4], GELU()),
                                     nn.Linear(layer_sizes[4], layer_sizes[5], GELU())])
        
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
                                       context_length=cfg["context_length"],
                                       num_heads=cfg["n_heads"],
                                       dropout=cfg["drop_rate"],
                                       qkv_bias=cfg["qkv_bias"])
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x): 
        shortcut = x
        x = self.norm1(x) 
        x = self.att(x)
        x = self.drop_shortcut(x) 
        x = x + shortcut

        shortcut = x
        x= self.norm2(x) 
        x = self.ff(x) 
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x 
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

 
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx): 
        batch_size, seg_len = in_idx.shape    
        tok_embeds = self.tok_emb(in_idx)     
        pos_embeds = self.pos_emb(torch.arange(seg_len, device=in_idx.device))   
        x = tok_embeds + pos_embeds  
        x = self.drop_emb(x)    
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x) 
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad(): 
            logits = model(idx_cond)
        logits = logits[:, -1, :] 
        probas = torch.softmax(logits, dim=-1) 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss / num_batches
        
GPT_CONFIG_124M = {
    "vocab_size":50257,     #词汇表大小
    "context_length":256,  #上下文长度       
    "emb_dim":3,          #嵌入维度
    "n_heads":3,           #注意力头的数量
    "n_layers":1,          #Transformer 层数量
    "drop_rate":0.1,        #dropout 率
    "qkv_bias":False,       #是否使用 qkv 偏置项
}

train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx: ]

torch.manual_seed(123)
train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
                                    stride=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True,
                                    num_workers=0)

val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
                                    stride=GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False,
                                    num_workers=0)

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)
start_context = "I am a"
encoded = tokenizer.encode(start_context);
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.eval()
out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=1, context_size=GPT_CONFIG_124M["context_length"])
# print("Output:", out)
# print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)


