# GPT-2 Internals — Study Notes

Reference repo: Andrej Karpathy's `build-nanogpt`
- `play.ipynb` — HuggingFace `GPT2LMHeadModel` exploration
- `train_gpt2.py` — from-scratch custom implementation

---

## 1. Token IDs and Vocabulary

Text is never fed raw into the model. A **tokenizer** (BPE — Byte Pair Encoding) splits text into subwords and maps each to an integer ID.

```
"Hello, world!" → [15496, 11, 995, 0]
```

GPT-2's vocabulary size is **50,257** tokens. Every unique subword has one ID in `[0, 50256]`.

---

## 2. Input Shape: (B, T)

```
B = Batch size  → number of sequences processed in parallel
T = Time steps  → number of tokens per sequence (max 1024 for GPT-2)
```

The raw input is a 2D integer tensor:

```python
input_ids.shape  # (4, 1024)

# Concretely:
[[31, 542, 17, 9, ...],   # sentence 1 token IDs
 [99,   3, 412, 7, ...],  # sentence 2
 [ 7,  18,  23, 1, ...],  # sentence 3
 [55, 200,   8, 4, ...]]  # sentence 4
```

Each cell is just an integer — a token ID. No meaning yet.

---

## 3. nn.Embedding — The Token Lookup Table

`nn.Embedding(vocab_size, embed_dim)` is a **matrix of shape (50257, 768)**.

Each row is a learned 768-number vector representing one token's meaning.

```python
wte = nn.Embedding(50257, 768)
# wte.weight.shape → (50257, 768)
# Row 31  → embedding for token "The"
# Row 542 → embedding for token "cat"
```

When you call `wte(input_ids)`, it **index-selects** rows — no matrix multiplication, just a lookup.

```
(B, T) integer tensor  →  (B, T, 768) float tensor
(4, 1024)              →  (4, 1024, 768)
```

Every token ID is replaced by its 768-number row.

---

## 4. What Is a 768-Dimensional Vector?

A vector is just a list of numbers. 768-dimensional means a list of 768 floats.

It represents the token's **meaning** as a point in 768-dimensional space. The model learns what each dimension encodes (grammar, topic, sentiment, etc.) — we don't assign meanings manually.

Simple 2D analogy:

```
"king"  → [0.9, 0.8]   # dim1 ≈ royalty, dim2 ≈ male
"queen" → [0.9, 0.2]
"man"   → [0.1, 0.8]
"woman" → [0.1, 0.2]
```

GPT-2 uses 768 dimensions so there's enough capacity to encode all aspects of language.

---

## 5. Positional Embeddings — wpe

Transformers have no built-in notion of order (unlike RNNs). Without positional info, "cat sat mat" and "mat cat sat" look identical.

**Solution:** a second embedding table `wpe = nn.Embedding(1024, 768)` — one learned vector per position (0 to 1023).

```python
pos = torch.arange(T)          # [0, 1, 2, ..., T-1]
pos_emb = wte(pos)             # (T, 768)
```

---

## 6. Why Positions Are ADDED, Not Concatenated

### Option A — Concatenate
```
token vector:    768 numbers  "what word am I"
position vector: 768 numbers  "where am I"
─────────────────────────────────────────────
combined:       1536 numbers
```
This works but **doubles** the vector size → 2× compute and parameters in every layer.

### Option B — Add (what GPT-2 uses)
```
token vector:    [t1, t2, ..., t768]
position vector: [p1, p2, ..., p768]
─────────────────────────────────────
sum:             [t1+p1, t2+p2, ..., t768+p768]  → still 768 numbers
```

**Why addition works:** 768 dimensions is a large enough space that token and position information can coexist without interfering. The attention layers downstream are full of learned linear projections that can *decode* both signals from the single combined vector.

Small analogy — encoding color and brightness in 3D:
```
red   = [1.0, 0.0, 0.0]
large = [0.5, 0.5, 0.5]

red + large = [1.5, 0.5, 0.5]  ← a linear layer can still separate color from size
```

In 768D there's vastly more room. Token meaning and position are kept in orthogonal-enough subspaces that the model learns to read both.

### Code (train_gpt2.py)
```python
tok_emb = self.transformer.wte(idx)   # (B, T, 768)
pos_emb = self.transformer.wpe(pos)   # (T, 768)   → broadcasts over B
x = tok_emb + pos_emb                 # (B, T, 768)
```

---

## 7. Full Embedding Pipeline Summary

```
Raw text
   ↓  tokenizer (BPE)
Token IDs          shape: (B, T)         dtype: int
   ↓  wte — token embedding lookup
Token embeddings   shape: (B, T, 768)    dtype: float
   +
Position embeddings shape: (T, 768)      ← one vector per slot 0..T-1
   ↓
Combined x         shape: (B, T, 768)    → fed into transformer blocks
```

Each of the B×T tokens is now a 768-float vector encoding **what word** + **where in sequence** before any attention has been applied.

---

## 8. HuggingFace vs Custom — Weight Names

| Concept | HuggingFace key | train_gpt2.py |
|---|---|---|
| Token embedding | `transformer.wte.weight` | `self.transformer.wte` |
| Position embedding | `transformer.wpe.weight` | `self.transformer.wpe` |
| Layer norm (pre-attn) | `transformer.h.0.ln_1.*` | `self.ln_1` |
| Attention QKV | `transformer.h.0.attn.c_attn.*` | `self.attn.c_attn` |
| MLP fc layer | `transformer.h.0.mlp.c_fc.*` | `self.mlp.c_fc` |
| Final layer norm | `transformer.ln_f.*` | `self.transformer.ln_f` |
| LM head | `lm_head.weight` | `self.lm_head` (tied to wte) |

---

*Notes cover: tokenization → token IDs → nn.Embedding → 768-dim vectors → positional embeddings → addition vs concat → full pipeline*
