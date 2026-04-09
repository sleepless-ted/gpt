import os
import torch
import torch.nn as nn
import setup
import math

input_file = os.path.join(setup.DATA_DIR, "input_clean.txt")

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# input_clean.txt is a space-separated sequence of word tokens.
tokens = text.split()
unique_tokens = sorted(set(tokens))
vocab_size = len(unique_tokens)

words_to_ids = {word: i for i, word in enumerate(unique_tokens)}
ids_to_words = {i: word for word, i in words_to_ids.items()}

encode = [words_to_ids[token] for token in tokens]
data = torch.tensor(encode, dtype=torch.long, device=setup.device)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # table d'embedding : vocab_size x embedding_dim. Pour la conversion de token en vecteur d'embedding
        self.word_token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim, device=setup.device)
        # table d'embedding : context_length x embedding_dim. Pour la conversion de position en vecteur d'embedding
        self.word_position_embedding_table = nn.Embedding(config.context_length, config.embedding_dim, device=setup.device)
        self.dropout_layer = nn.Dropout(config.dropout)
        # stack des blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)])
        self.norm_layer = nn.LayerNorm(config.embedding_dim)
        self.llm_projection_layer = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Decode words from the embedding space using the same weights as the token encoding table
        self.llm_projection_layer.weight = self.word_token_embedding_table.weight

    def forward(self, idx, targets=None):
        token_embeddings = self.word_token_embedding_table(idx)
        position_embeddings = self.word_position_embedding_table(torch.arange(idx.size(1), device=setup.device))
        X = token_embeddings + position_embeddings
        X = X.to(setup.device)
        X = self.dropout_layer(X)
        for block in self.blocks:
            X = block(X)
        X = self.norm_layer(X)
        logits = self.llm_projection_layer(X)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # crop idx to the last context_length tokens
            idx_crop = idx[:, -self.config.context_length:]
            logits, _ = self(idx_crop)
            # increasing temperature will make the model more "creative" by increasing the probability of less likely tokens
            logits = logits[:, -1, :] / temperature
            probs = nn.functional.softmax(logits, dim=-1)
            # sample the next token from a miltinomial distribution
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.embedding_dim % config.num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.key_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.query_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.value_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        # lower triangular matrix to mask out future tokens in the attention mechanism
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
        )

    def forward(self, x):
        batch, seq, embd_dim = x.size()
        head_dim = embd_dim // self.config.num_heads
        mask = self.mask[:seq, :seq]

        K = self.key_projection(x) # (batch, seq, embd_dim)
        Q = self.query_projection(x) # (batch, seq, embd_dim)
        V = self.value_projection(x) # (batch, seq, embd_dim)
        
        K = K.view(batch, seq, self.config.num_heads, head_dim).transpose(1, 2) # (batch, num_heads, seq, head_dim)
        Q = Q.view(batch, seq, self.config.num_heads, head_dim).transpose(1, 2) # (batch, num_heads, seq, head_dim)
        V = V.view(batch, seq, self.config.num_heads, head_dim).transpose(1, 2) # (batch, num_heads, seq, head_dim)
        
        # compute attention scores (or affinities) between tokens 
        W = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
        W = W.masked_fill(mask == 0, float('-inf'))
        W = nn.functional.softmax(W, dim=-1)
        Y = W @ V
        
        # reshape back to (batch, seq, num_heads, head_dim)
        Y = Y.transpose(1, 2).contiguous()
        # Concatenate the heads to get back to (batch, seq, embd_dim)
        Y = Y.view(batch, seq, embd_dim)
        # Project back to the original embedding dimension
        Y = self.output_projection(Y)

        return Y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expension_layer = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.projection_layer = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, X):
        X = self.expension_layer(X)
        X = nn.functional.gelu(X)
        X = self.projection_layer(X)
        X = self.dropout_layer(X)
        return X

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_layer_1 = nn.LayerNorm(config.embedding_dim)
        self.norm_layer_2 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, X):
        # residual connection to prevent vanishing gradients
        X = X + self.attention(self.norm_layer_1(X))
        X = X + self.mlp(self.norm_layer_2(X))
        return X
    
    
    
    
    
    
    
    
    
    
    
    