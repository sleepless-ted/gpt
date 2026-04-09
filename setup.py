import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, 'data')

device = 'cuda'

class GPTConfig:
    vocab_size : int = 50000 # nombre de tokens à garder dans le vocabulaire (les plus fréquents)
    embedding_dim : int = 128 # dimension de l'espace d'embedding
    batch_size : int = 32
    context_length : int = 64 # nombre de tokens dans une séquence
    num_heads : int = 4
    head_dim = embedding_dim // num_heads
    dropout : float = 0.1
    num_blocks : int = 4
    
class TrainingConfig:
    data_length : int = 5000000 # nombre de tokens à utiliser pour l'entraînement   
    train_split : float = 0.9
    eval_iters : int = 200
    learning_rate : float = 3e-4
    weight_decay : float = 1e-2
    steps : int = 1000
    eval_interval : int = 100
    checkpoint : str = "checkpoint.pt"
    sample_tokens : int = 1000