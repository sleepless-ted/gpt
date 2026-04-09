import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, 'data')

device = 'cuda'

class GPTConfig:
    vocab_size : int = 50000 # nombre de tokens à garder dans le vocabulaire (les plus fréquents)
    embedding_dim : int = 128 # dimension de l'espace d'embedding
    batch_size : int = 64
    context_length : int = 1024 # nombre de tokens dans une séquence