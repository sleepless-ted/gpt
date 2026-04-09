import os
import torch
import torch.nn as nn
import setup

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

    def forward(self, idx):
        token_embeddings = self.word_token_embedding_table(idx)
        position_embeddings = self.word_position_embedding_table(torch.arange(idx.size(1), device=setup.device))
        return token_embeddings + position_embeddings
    
    
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False, device=setup.device)
        self.query_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False, device=setup.device)
        self.value_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False, device=setup.device)
        
    def forward(self, x):
        batch, seq, embd_dim = x.size()
        K = self.key_projection(x)
        Q = self.query_projection(x)
        V = self.value_projection(x)

        # compute attention scores (or affinities) between tokens 
        W = Q @ K.transpose(-2, -1) / embd_dim**0.5
        W = torch.nn.functional.softmax(W, dim=-1) 

        out = W @ V
        return out