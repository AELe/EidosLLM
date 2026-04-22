import torch
import torch.nn as nn

# 从你的代码中复制 MultiHeadAttention 类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
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
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


print("="*80)
print("多头注意力机制 - 一针见血版")
print("="*80)

# 你的例子
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # journey
    [0.57, 0.85, 0.64],  # starts
    [0.22, 0.58, 0.33],  # with
    [0.77, 0.25, 0.10],  # one
    [0.05, 0.80, 0.55],  # step
])

batch = torch.stack([inputs, inputs], dim=0)
torch.manual_seed(123)

d_in = 3
d_out = 2
num_heads = 2
head_dim = 1

mha = MultiHeadAttention(d_in, d_out, 6, 0.0, num_heads=2)

print("\n【核心问题】")
print("句子：\"Your journey starts with one step\"")
print(f"输入：6 个 token，每个{d_in}维")
print(f"输出：6 个 token，每个{d_out}维")
print(f"但用了{num_heads}个注意力头，每个头{head_dim}维")
print("\n为什么这么设计？看下面...")

# 前向传播，逐步展示
b, num_tokens, _ = batch.shape
queries = mha.W_query(batch)
keys = mha.W_key(batch)
values = mha.W_value(batch)

print("\n" + "="*80)
print("【第一步：QKV 投影】")
print("="*80)
print(f"\nqueries.shape = {queries.shape}")
print(f"\nqueries[0] (第一个样本的所有 token):")
print(queries[0].detach())

print("\n【关键】")
print("每个 token 从 3 维 → 2 维")
print("这 2 维会被拆分成 2 个头，每个头 1 维")
print("也就是说：每个 token 有 2 个不同的表示角度")

# 拆分多头
keys_view = keys.view(b, num_tokens, num_heads, head_dim)
queries_view = queries.view(b, num_tokens, num_heads, head_dim)
values_view = values.view(b, num_tokens, num_heads, head_dim)

keys_trans = keys_view.transpose(1, 2)
queries_trans = queries_view.transpose(1, 2)
values_trans = values_view.transpose(1, 2)

print("\n" + "="*80)
print("【第二步：多头拆分】")
print("="*80)
print(f"\n拆分后：keys_trans.shape = {keys_trans.shape}")
print(f"含义：(batch=2, heads=2, tokens=6, head_dim=1)")

print("\n【关键】")
print("现在每个 token 有 2 个独立的表示：")
print("  head 0: queries_trans[0, 0, :, 0] =", queries_trans[0, 0, :, 0].detach().tolist())
print("  head 1: queries_trans[0, 1, :, 0] =", queries_trans[0, 1, :, 0].detach().tolist())
print("\nhead 0 和 head 1 会独立计算注意力！")

# 计算注意力
attn_scores = queries_trans @ keys_trans.transpose(2, 3)

print("\n" + "="*80)
print("【第三步：两个头独立计算注意力】")
print("="*80)
print(f"\nattn_scores.shape = {attn_scores.shape}")
print(f"含义：2 个 batch，2 个头，每个头计算 6×6 的注意力矩阵")

print("\n【head 0 的注意力分数】")
print(attn_scores[0, 0].detach())
print("\n【head 1 的注意力分数】")
print(attn_scores[0, 1].detach())

print("\n【关键观察】")
print("head 0 和 head 1 的注意力分数是**不同**的！")
print("这意味着：两个头关注不同的 token 关系")
print(f"\nhead 0[0,0] = {attn_scores[0, 0, 0, 0].detach().item():.4f}")
print(f"head 1[0,0] = {attn_scores[0, 1, 0, 0].detach().item():.4f}")

# 应用掩码
mask_bool = mha.mask.bool()[:num_tokens, :num_tokens]
attn_scores_masked = attn_scores.clone()
attn_scores_masked.masked_fill_(mask_bool, float('-inf'))

attn_weights = torch.softmax(attn_scores_masked / head_dim**0.5, dim=-1)

print("\n" + "="*80)
print("【第四步：两个头的注意力权重】")
print("="*80)

print(f"\n【head 0 的注意力权重 (只看前 3 个 token)】")
print("token 0 (Your):", attn_weights[0, 0, 0, :3].detach().tolist())
print("token 1 (journey):", attn_weights[0, 0, 1, :3].detach().tolist())
print("token 2 (starts):", attn_weights[0, 0, 2, :3].detach().tolist())

print(f"\n【head 1 的注意力权重 (只看前 3 个 token)】")
print("token 0 (Your):", attn_weights[0, 1, 0, :3].detach().tolist())
print("token 1 (journey):", attn_weights[0, 1, 1, :3].detach().tolist())
print("token 2 (starts):", attn_weights[0, 1, 2, :3].detach().tolist())

print("\n【关键】")
print("head 0 和 head 1 的注意力权重**不同**！")
print("说明：两个头在学习**不同**的注意力模式")
print("  - head 0 可能关注语法关系")
print("  - head 1 可能关注语义关系")
print(f"\nhead 0 权重 [0,0,:] = {attn_weights[0, 0, 0, :3].detach().tolist()}")
print(f"head 1 权重 [0,1,:] = {attn_weights[0, 1, 0, :3].detach().tolist()}")

# 加权求和
context_vec = attn_weights @ values_trans

print("\n" + "="*80)
print("【第五步：两个头独立加权求和】")
print("="*80)
print(f"\ncontext_vec.shape = {context_vec.shape}")
print(f"含义：每个头独立计算出一个上下文向量")

print(f"\n【head 0 计算的上下文向量 (前 3 个 token)】")
print("token 0 (Your):", context_vec[0, 0, 0, 0].detach().item())
print("token 1 (journey):", context_vec[0, 0, 1, 0].detach().item())
print("token 2 (starts):", context_vec[0, 0, 2, 0].detach().item())

print(f"\n【head 1 计算的上下文向量 (前 3 个 token)】")
print("token 0 (Your):", context_vec[0, 1, 0, 0].detach().item())
print("token 1 (journey):", context_vec[0, 1, 1, 0].detach().item())
print("token 2 (starts):", context_vec[0, 1, 2, 0].detach().item())

print("\n【关键】")
print("同一个 token，head 0 和 head 1 算出的上下文向量**不同**！")
print("因为：两个头的注意力权重不同，加权求和的结果就不同")
print(f"\nhead 0 的 context_vec[0,0,0,0] = {context_vec[0, 0, 0, 0].detach().item():.4f}")
print(f"head 1 的 context_vec[0,1,0,0] = {context_vec[0, 1, 0, 0].detach().item():.4f}")

# 转置 + 拼接
context_vec_trans = context_vec.transpose(1, 2)
context_vec_concat = context_vec_trans.contiguous().view(b, num_tokens, d_out)

print("\n" + "="*80)
print("【第六步：拼接两个头的结果】")
print("="*80)
print(f"\n拼接后：context_vec_concat.shape = {context_vec_concat.shape}")

print(f"\n【每个 token 的最终表示 = head0 + head1 拼接】")
for i, word in enumerate(["Your", "journey", "starts", "with", "one", "step"]):
    vec = context_vec_concat[0, i].detach().tolist()
    print(f"{word:8s}: [{vec[0]:7.4f}, {vec[1]:7.4f}]")
    if i < 3:
        print(f"           ↑head0     ↑head1")

print("\n【关键】")
print("每个 token 的 2 维向量：")
print("  - 第 1 维来自 head 0 的计算结果")
print("  - 第 2 维来自 head 1 的计算结果")
print("  - 两个头从不同角度理解这个 token")

# 输出投影
output = mha.out_proj(context_vec_concat)

print("\n" + "="*80)
print("【第七步：输出投影（整合两个头）】")
print("="*80)
print(f"\n最终输出：output.shape = {output.shape}")

print(f"\n【最终结果】")
for i, word in enumerate(["Your", "journey", "starts", "with", "one", "step"]):
    vec = output[0, i].detach().tolist()
    print(f"{word:8s}: [{vec[0]:7.4f}, {vec[1]:7.4f}]")

print("\n" + "="*80)
print("【醍醐灌顶总结】")
print("="*80)
print("""
问题：为什么要把 d_out=2 拆成 2 个头，每个头 1 维？

答案：
1️⃣ 如果不用多头（只有 1 个头，2 维）：
   - 每个 token 只能用一种方式计算注意力
   - 只能学到一种注意力模式

2️⃣ 用了多头（2 个头，每个 1 维）：
   - head 0 用 1 维计算一次注意力 → 得到 1 个上下文向量
   - head 1 用 1 维计算一次注意力 → 得到另 1 个上下文向量
   - 两个头独立学习，关注不同的模式
   - 最后拼接：[head0_result, head1_result]

3️⃣ 核心优势：
   - 多头 = 多个不同的"视角"
   - head 0 可能关注："Your" 和 "journey" 的语法关系（所有格）
   - head 1 可能关注："journey" 和 "starts" 的语义关系（主谓）
   - 拼接后同时保留两种信息

4️⃣ 你的例子中：
   每个 token 输出 2 维 = [head0 的视角，head1 的视角]
   
   "Your": [0.32, 0.49]
           ↑head0 认为    ↑head1 认为
           "Your"重要     "Your"重要
   
   两个头独立计算，然后拼接，让模型能从多个角度理解每个 token！

这就是多头的本质：用多个独立的小头，学习多种注意力模式，
然后拼接起来，得到更丰富的表示。
""")

print("="*80)
