import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a single batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and not config.no_flash_attn
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # KV Cache interaction
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        if use_cache:
            present_kv = (k, v)
        else:
            present_kv = None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # using flash attention if available
        # logic for manual attention needs update for past_kv size mismatch in mask
        # but pure Flash Attention usually handles this or we rely on explicit masking
        # For simplicity with implementation, let's use manual implementation if past_kv is involved 
        # OR just rely on sdpa handling it if we pass correct mask.
        # But F.scaled_dot_product_attention expects q, k, v. 
        # If k,v are longer than q, it computes attention correctly for q against all k,v.
        # We just need to ensure causal mask is correct. 
        # For inference (q len 1, k len N), is_causal=False is fine as we attend to all past.
        
        if self.flash:
            # flash attention
            # strict causal masking is handled by is_causal=True. 
            # However, if T_query != T_key, is_causal=True in sdpa might assume diagonal alignment which is tricky.
            # Usually for decoding (T_q=1), is_causal=False is fine as we attend to all past.
            is_causal = True if past_kv is None else False
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        else:
            # manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # If past_kv is present, T is the current query length (usually 1).
            # The bias mask needs to be adjusted for the full sequence length (past_length + T).
            # This part of the manual implementation is tricky with past_kv.
            # For simplicity, if past_kv is used, we assume T=1 and no causal mask is needed for the single query token.
            # If T > 1 and past_kv is None, then the original causal mask applies.
            if past_kv is None:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # If past_kv is not None, and T=1, we don't need to mask, as the single query token attends to all past.
            # If T > 1 and past_kv is not None, this manual implementation would need more complex masking.
            # For now, we assume T=1 for generation with past_kv.
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        if config.num_classes is not None:
            self.transformer['wce'] = nn.Embedding(config.num_classes, config.n_embd)
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying, the lm_head.weight is shared with wte.weight
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            if 'wce' in self.transformer:
                n_params -= self.transformer.wce.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, class_labels=None, past_kvs=None, use_cache=False, all_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # positional embeddings will depend on whether we have past_kvs (inference)
        if past_kvs is not None:
            # We are generating the next token, idx is (B, 1) usually
            # The position is the length of past
            past_length = past_kvs[0][0].size(-2) # (B, nh, T_past, hs)
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        # Combine embeddings
        if class_labels is not None and self.config.num_classes is not None:
            # Global class conditioning
            # We add it at every token? Or just start?
            # Standard conditional GAN/GPT style: add to embedding
            cls_emb = self.transformer.wce(class_labels) # (b, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb + cls_emb.unsqueeze(1))
        else:
            x = self.transformer.drop(tok_emb + pos_emb)
            
        new_past_kvs = [] if use_cache else None
        
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_past_kvs.append(present_kv)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            if all_logits:
                logits = self.lm_head(x)
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, new_past_kvs # Updated signature logic (compatible with old calls if unwrapped carefully, but we should check) 

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we're doing fine-tuning on a model trained with a larger block size
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, class_labels=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        past_kvs = None
        for k in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # For the first token, idx_cond is the full input. For subsequent tokens, it's just the last token.
            if k == 0: # First token generation, process full context
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else: # Subsequent tokens, only pass the last generated token
                idx_cond = idx[:, -1:]

            # forward the model to get the logits for the index in the sequence
            logits, _, past_kvs = self(idx_cond, class_labels=class_labels, past_kvs=past_kvs, use_cache=True)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, num_classes=None, dropout=0.0, no_flash_attn=False, bias=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.num_classes = num_classes
        self.dropout = dropout
        self.no_flash_attn = no_flash_attn
        self.bias = bias
