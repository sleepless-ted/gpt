import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, 'data')

device = 'cuda'

class GPTConfig:
    vocab_size : int = 50000 # nombre de tokens à garder dans le vocabulaire (les plus fréquents)
    embedding_dim : int = 32 * 12 # dimension de l'espace d'embedding
    batch_size : int = 64
    context_length : int = 128 # nombre de tokens dans une séquence
    num_heads : int = 12
    head_dim = embedding_dim // num_heads
    dropout : float = 0.1
    num_blocks : int = 5
    
class TrainingConfig:
    data_length : int = 2000000 # nombre de tokens à utiliser pour l'entraînement   
    train_split : float = 0.9
    learning_rate : float = 3e-4
    weight_decay : float = 1e-2
    steps : int = 20000 # nombre d'itérations d'entraînement
    eval_iters : int = 100 # nombre d'itérations pour estimer la perte moyenne
    eval_interval : int = 500 # nombre d'itérations entre chaque évaluation de la perte
    checkpoint : str = "checkpoint.pt"
    sample_tokens : int = 1000