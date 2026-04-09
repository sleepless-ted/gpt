import os
import torch
import setup
from model import GPT


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dataset(config):
    input_file = os.path.join(setup.DATA_DIR, "input_clean.txt")
    with open(input_file, "r", encoding="utf-8") as f:
        tokens = f.read().split()

    data_length = min(config.data_length, len(tokens))
    tokens = tokens[:data_length].copy()  # limit dataset size for faster training during development
    unique_tokens = sorted(set(tokens))
    words_to_ids = {word: i for i, word in enumerate(unique_tokens)}
    ids_to_words = {i: word for word, i in words_to_ids.items()}

    config.vocab_size = len(unique_tokens)
    encoded = torch.tensor([words_to_ids[token] for token in tokens], dtype=torch.long, device=device)

    split_idx = int(len(encoded) * config.train_split)
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    return train_data, val_data, words_to_ids, ids_to_words


def get_batch(split_data, gpt_config):
    max_start = len(split_data) - gpt_config.context_length - 1
    positions = torch.randint(0, max_start + 1, (gpt_config.batch_size,))
    x = torch.stack([split_data[pos:pos + gpt_config.context_length] for pos in positions])
    y = torch.stack([split_data[pos + 1:pos + gpt_config.context_length + 1] for pos in positions])
    return x.to(device), y.to(device)


@torch.inference_mode()
def estimate_loss(model, train_data, val_data, training_config, gpt_config):
    losses = {}
    model.eval()
    for split_name, split_data in (("train", train_data), ("val", val_data)):
        split_losses = torch.zeros(training_config.eval_iters)
        for i in range(training_config.eval_iters):
            x, y = get_batch(split_data, gpt_config)
            _, loss = model(x, y)
            split_losses[i] = loss.item()
        losses[split_name] = split_losses.mean().item()
    model.train()
    return losses


def decode_tokens(token_ids, ids_to_words):
    return " ".join(ids_to_words[token_id] for token_id in token_ids)


def save_checkpoint(path, model, optimizer, gpt_config, training_config, words_to_ids, ids_to_words):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": gpt_config.vocab_size,
            "embedding_dim": gpt_config.embedding_dim,
            "batch_size": gpt_config.batch_size,
            "context_length": gpt_config.context_length,
            "num_heads": gpt_config.num_heads,
            "dropout": gpt_config.dropout,
            "num_blocks": gpt_config.num_blocks,
        },
        "training_config": {
            "train_split": training_config.train_split,
            "eval_iters": training_config.eval_iters,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "steps": training_config.steps,
            "eval_interval": training_config.eval_interval,
            "checkpoint": training_config.checkpoint,
        },
        "words_to_ids": words_to_ids,
        "ids_to_words": ids_to_words,
    }
    torch.save(checkpoint, path)


def main():
    global device
    device = resolve_device()
    gpt_config = setup.GPTConfig()
    training_config = setup.TrainingConfig()
    
    train_data, val_data, words_to_ids, ids_to_words = load_dataset(training_config)

    model = GPT(gpt_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    print(f"Training on {device}")
    print(f"Vocabulary size: {gpt_config.vocab_size}")
    print(f"Train tokens: {len(train_data)} | Validation tokens: {len(val_data)}")

    for step in range(training_config.steps):
        if step % training_config.eval_interval == 0 or step == training_config.steps - 1:
            losses = estimate_loss(model, train_data, val_data, training_config, gpt_config)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, gpt_config)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint_path = os.path.abspath(training_config.checkpoint)
    save_checkpoint(checkpoint_path, model, optimizer, gpt_config, training_config, words_to_ids, ids_to_words)
    print(f"Saved checkpoint to {checkpoint_path}")

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=training_config.sample_tokens)[0].tolist()
    print("Sample generation:")
    print(decode_tokens(generated, ids_to_words))


if __name__ == "__main__":
    main()