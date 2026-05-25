# GPT-2 Block 1 — CausalSelfAttention + MLP
*Date: 2026-05-25*

---

## Quick Revision — Where We Are

After the embedding pipeline, `x` has shape `(B, T, 768)`. It flows through 12 identical blocks:

```python
# train_gpt2.py:120-121
for block in self.transformer.h:
    x = block(x)
```

Each block does two things — **attention** then **MLP** — each wrapped in LayerNorm + residual:

```python
# train_gpt2.py:66-68
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # Part 1: attention
    x = x + self.mlp(self.ln_2(x))    # Part 2: MLP
```

---

## Why Attention Exists

After embedding, every token's 768-vector only knows about **itself**. Attention lets tokens **talk to each other** — after attention, the vector for "cat" can encode "black cat" because it has seen and incorporated info from surrounding tokens.

> **Forward pass:** One complete run of input data through the entire model pipeline to produce an output. Training does forward pass + backward pass (weight update). Inference does forward pass only.

---

## Part 1 — CausalSelfAttention

### Step 1 — QKV Projection (`c_attn`)

**play.ipynb state dict:**
```
transformer.h.0.attn.c_attn.weight   torch.Size([768, 2304])
transformer.h.0.attn.c_attn.bias     torch.Size([2304])
```

**train_gpt2.py:18:**
```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
# nn.Linear(768, 2304)   because 2304 = 3 × 768
```

One linear layer projects 768 → 2304, then split into Q, K, V:

```python
# train_gpt2.py:31-32
qkv = self.c_attn(x)                       # (B, T, 768) → (B, T, 2304)
q, k, v = qkv.split(self.n_embd, dim=2)   # each (B, T, 768)
```

> **nn.Linear:** Matrix multiply + bias. Input (768) × weight matrix (768×2304) = output (2304). Every output number is a learned weighted sum of all 768 input numbers.

**What are Q, K, V?**

| | Library analogy | In attention |
|---|---|---|
| **Q (Query)** | "I want books about cats" | What this token is looking for |
| **K (Key)** | Title/tag on each book spine | What each token advertises about itself |
| **V (Value)** | Actual book content | Information a token passes along if selected |

---

### Step 2 — Splitting into Heads

GPT-2 uses 12 attention heads. 768 dims split into 12 × 64.

```python
# train_gpt2.py:33-35
# C=768, self.n_head=12, C//self.n_head=64
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, 12, T, 64)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, 12, T, 64)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, 12, T, 64)
```

> **`.view()`:** Reshapes a tensor without changing data. `(B, T, 768)` → `(B, T, 12, 64)` — same numbers, refolded.
> **`.transpose(1,2)`:** Swaps T and head dims → `(B, 12, T, 64)`. Each head gets its own independent `(B, T, 64)` tensor.

**Why multiple heads?** Each head learns a different type of relationship — subject-verb agreement, coreference ("it" → "cat"), positional proximity, etc. 12 small attentions in parallel = more expressive than one big one.

---

### Step 3 — Scaled Dot-Product Attention

```python
# train_gpt2.py:36
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, 12, T, 64)
```

Four things happen inside:

**3a. Dot product scores**
```
scores = Q @ K^T          # (B, 12, T, T)
# scores[b, h, i, j] = how much token i wants to attend to token j
```

> **Dot product:** Multiply two vectors element-wise, then sum. High result = vectors compatible. `[1,0,1]·[1,0,1]=2` (high), `[1,0,0]·[0,1,0]=0` (unrelated).

**3b. Scale by √64**
```
scores = scores / sqrt(64)   # ÷ 8
```
Prevents large dot products from saturating softmax and killing gradients.

**3c. Causal mask (`is_causal=True`)**

Token at position i cannot see tokens at positions j > i (the future):
```
scores[i, j] = -inf   if j > i    (future — blocked)
scores[i, j] = raw    if j <= i   (past/current — allowed)
```
This is what makes it a *language model* — it predicts left to right.

**3d. Softmax → weighted sum of V**
```
weights = softmax(scores)    # (B, 12, T, T) — each row sums to 1
y = weights @ V              # (B, 12, T, 64)
```

> **Softmax:** Converts raw scores to probabilities summing to 1. `softmax([2.0, 1.0, 0.1]) → [0.71, 0.26, 0.03]`.

---

### Step 4 — Reassemble Heads + Output Projection

```python
# train_gpt2.py:37-39
y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, 12, T, 64) → (B, T, 768)
y = self.c_proj(y)                                  # (B, T, 768) → (B, T, 768)
```

**play.ipynb:**
```
transformer.h.0.attn.c_proj.weight   torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias     torch.Size([768])
```

12 × 64 = 768 glued back together. `c_proj` mixes outputs across all heads.

**Attention shape trace:**
```
x in:              (B, T, 768)
  c_attn           (B, T, 2304)
  split Q,K,V      each (B, T, 768)
  view+transpose   each (B, 12, T, 64)
  scaled_dot_prod  (B, 12, T, 64)
  transpose+view   (B, T, 768)
  c_proj           (B, T, 768)
attention out:     (B, T, 768)
```

---

## Part 2 — MLP

Attention lets tokens gather context. MLP then processes each token **independently** — per-token feature refinement / memory lookup.

**play.ipynb:**
```
transformer.h.0.mlp.c_fc.weight     torch.Size([768, 3072])   # 768 → 4×768
transformer.h.0.mlp.c_fc.bias       torch.Size([3072])
transformer.h.0.mlp.c_proj.weight   torch.Size([3072, 768])   # 4×768 → 768
transformer.h.0.mlp.c_proj.bias     torch.Size([768])
```

**train_gpt2.py:42-55:**
```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)  # 768 → 3072
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # 3072 → 768

    def forward(self, x):
        x = self.c_fc(x)    # (B, T, 768) → (B, T, 3072)   expand
        x = self.gelu(x)    # (B, T, 3072)                  non-linearity
        x = self.c_proj(x)  # (B, T, 3072) → (B, T, 768)   contract
```

Pattern: **expand → activate → contract**. 4× expansion gives a wider space to represent complex feature combinations before projecting back.

> **GELU (activation function):** Introduces non-linearity. Without it, stacking linear layers = mathematically still just one linear layer. GELU: values < -3 → ≈ 0 (blocked), values > 3 → passed through, in between → smooth curve. Smoother than ReLU which hard-zeros all negatives.

---

## Complete Block Shape Trace

```
x in:            (B, T, 768)
  ln_1           (B, T, 768)   normalize
  attn           (B, T, 768)   tokens gather context from each other
  + x residual   (B, T, 768)
  ln_2           (B, T, 768)   normalize
  mlp            (B, T, 768)   per-token feature refinement
  + x residual   (B, T, 768)
x out:           (B, T, 768)   same shape, richer representation
```

Repeats 12 times → then `ln_f` → then `lm_head` → word predictions.

---

## Key Weight Names (HF ↔ Custom)

| Component | play.ipynb (HuggingFace) | train_gpt2.py |
|---|---|---|
| LayerNorm 1 | `transformer.h.0.ln_1.weight/bias` | `self.ln_1` |
| QKV projection | `transformer.h.0.attn.c_attn.weight/bias` | `self.attn.c_attn` |
| Attn output proj | `transformer.h.0.attn.c_proj.weight/bias` | `self.attn.c_proj` |
| LayerNorm 2 | `transformer.h.0.ln_2.weight/bias` | `self.ln_2` |
| MLP expand | `transformer.h.0.mlp.c_fc.weight/bias` | `self.mlp.c_fc` |
| MLP contract | `transformer.h.0.mlp.c_proj.weight/bias` | `self.mlp.c_proj` |
