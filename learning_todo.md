# GPT-2 Learning TODO

> Method: paste code snippets from `play.ipynb` (HuggingFace) and `train_gpt2.py` (custom), compare side-by-side, explain theory + examples.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## PART 1 — Foundations & Embeddings
- [x] **Tokenization** — BPE, tiktoken, vocab size 50257, encode/decode
- [x] **Input shape (B, T)** — batch size, sequence length, integer token IDs
- [x] **nn.Embedding / wte** — lookup table (50257 × 768), index-select, shape (B,T)→(B,T,768)
- [x] **768-dimensional vectors** — what embedding dimension means, analogy
- [x] **Positional embeddings / wpe** — why needed, nn.Embedding(1024, 768)
- [x] **Token + Position: add vs concat** — why addition works in high-dim space
- [ ] **x / y target construction** — `buf[:-1]` and `buf[1:]`, why shifted by 1 (next-token prediction)
- [ ] **Weight tying: wte == lm_head** — same tensor, why it works, memory pointer proof from play.ipynb

---

## PART 2 — Normalization & Activations
- [ ] **LayerNorm (ln_1, ln_2, ln_f)** — what it normalizes, gamma/beta, why before attention (pre-norm vs post-norm)
- [ ] **GELU activation** — shape of curve, why not ReLU, `approximate='tanh'` variant

---

## PART 3 — Attention
- [ ] **QKV projections** — `c_attn: Linear(768, 3×768)`, splitting into Q, K, V
- [ ] **Multi-head split** — reshape to `(B, nh, T, hs)`, 12 heads × 64 head-size = 768
- [ ] **Scaled dot-product attention** — Q·Kᵀ / √dₖ, softmax, weighted sum of V
- [ ] **Causal masking** — why upper triangle is masked (can't see future tokens)
- [ ] **Flash Attention** — `F.scaled_dot_product_attention(is_causal=True)`, what makes it fast
- [ ] **Output projection (c_proj)** — reassemble heads, project back to 768
- [ ] **Attention weight visualization** — `h.1.attn.c_attn.weight[:300,:300]` from play.ipynb

---

## PART 4 — MLP Block
- [ ] **MLP structure** — Linear(768→3072) → GELU → Linear(3072→768), 4× expansion
- [ ] **Why 4× expansion** — capacity for "knowledge storage", FFN as key-value memory
- [ ] **NANOGPT_SCALE_INIT** — scaled weight init for residual projections, why `(2 × n_layer)^-0.5`

---

## PART 5 — Transformer Block & Full Model
- [ ] **Residual connections** — `x = x + attn(ln_1(x))`, why they help (gradient highway), std dev growth demo from play.ipynb
- [ ] **Block structure** — pre-norm, attn residual, mlp residual
- [ ] **GPTConfig** — all hyperparameters, 4 model sizes (124M → 1.5B)
- [ ] **GPT class assembly** — wte, wpe, 12× Block, ln_f, lm_head
- [ ] **Weight initialization** — `std=0.02` normal, zeros bias, scaled residual
- [ ] **Full forward pass** — embeddings → 12 blocks → ln_f → lm_head → logits `(B,T,50257)` → cross_entropy loss
- [ ] **from_pretrained** — loading HF weights, Conv1D vs Linear (why weights are transposed)

---

## PART 6 — Training Mechanics
- [ ] **Cross-entropy loss** — logits to probabilities, `view(-1, vocab_size)`, what the loss value means
- [ ] **AdamW optimizer** — momentum + adaptive LR, weight decay, β₁=0.9 β₂=0.95
- [ ] **Weight decay groups** — 2D params (matmuls) decay, 1D params (biases, layernorms) don't, why
- [ ] **Fused AdamW** — what "fused" means, why faster on CUDA
- [ ] **Cosine LR schedule** — linear warmup (715 steps) + cosine decay to min_lr, `get_lr()` walkthrough
- [ ] **Gradient clipping** — `clip_grad_norm_(1.0)`, why it prevents explosions
- [ ] **bfloat16 autocast** — mixed precision, why bfloat16 over float16 for training
- [ ] **Gradient accumulation** — micro-steps, loss scaling (`loss / grad_accum_steps`), math demo from play.ipynb
- [ ] **`torch.set_float32_matmul_precision('high')`** — TF32 on A100, what it trades off

---

## PART 7 — Data Pipeline
- [ ] **DataLoaderLite** — shard files, `next_batch()`, position tracking
- [ ] **DDP-aware data loading** — each rank reads different slice: `position = B * T * process_rank`
- [ ] **EduFineWeb-10B dataset** — what it is, why used instead of raw text

---

## PART 8 — Generation
- [ ] **Logits → probabilities** — `softmax(logits[:, -1, :])`, why only last position
- [ ] **Top-k sampling** — `torch.topk(probs, 50)`, why top-50, trade-off vs greedy
- [ ] **Multinomial sampling** — sampling from distribution vs argmax
- [ ] **Autoregressive loop** — append token, re-run forward, grow sequence

---

## PART 9 — Evaluation & Multi-GPU
- [ ] **Validation loss** — 20-step average, eval mode, `torch.no_grad()`
- [ ] **HellaSwag benchmark** — what it tests, how `get_most_likely_row()` works
- [ ] **DDP (Distributed Data Parallel)** — torchrun, RANK/LOCAL_RANK/WORLD_SIZE, `dist.all_reduce`
- [ ] **Checkpointing** — what's saved, how to resume
- [ ] **Training log visualization** — loss/accuracy curves from play.ipynb

---

## PART 10 — Visualizations (play.ipynb specific)
- [ ] **wpe weight matrix visualization** — `plt.imshow(wpe.weight)`, what patterns appear
- [ ] **wpe column plots** — periodic structure in positional embeddings
- [ ] **c_attn weight visualization** — `h.1.attn.c_attn.weight[:300,:300]`

---

*Progress: 6 / ~45 topics done*
