"""Interactive Gradio tour of the small GPT model in this repository."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

import setup
from model import GPT

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")
TOKEN_PATTERN = r"[a-z]+'[a-z]+|[a-z0-9]+(?:\.[0-9]+)?|[.,!?;:]"


def split_tokens(text: str) -> list[str]:
    """Tokenise like the training corpus without requiring NLTK data files."""
    return re.findall(TOKEN_PATTERN, (text or "").lower())


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_path() -> str | None:
    """Use the local artifact, or optionally fetch it from a HF model repo."""
    if os.path.exists(CHECKPOINT):
        return CHECKPOINT
    repo_id = os.environ.get("MODEL_REPO")
    if not repo_id:
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo_id,
            filename=os.environ.get("MODEL_FILENAME", "checkpoint.pt"),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise RuntimeError(f"téléchargement depuis MODEL_REPO impossible: {exc}") from exc


@lru_cache(maxsize=1)
def load_model() -> tuple[GPT | None, dict[str, Any] | None, str]:
    """Load the checkpoint once and map it to the available device."""
    try:
        checkpoint_path = _checkpoint_path()
        if checkpoint_path is None:
            return None, None, "Le fichier checkpoint.pt est absent (ou configure MODEL_REPO)."
        device = _device()
        setup.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = SimpleNamespace(**checkpoint["config"])
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        metadata = {
            "config": config,
            "words_to_ids": checkpoint["words_to_ids"],
            "ids_to_words": checkpoint["ids_to_words"],
            "device": device,
        }
        return model, metadata, "ok"
    except Exception as exc:
        return None, None, f"Impossible de charger le checkpoint : {type(exc).__name__}: {exc}"


def tokenise(text: str) -> tuple[str, pd.DataFrame, str]:
    tokens = split_tokens(text)
    _, metadata, status = load_model()
    known = metadata["words_to_ids"] if metadata else {}
    rows = [{"position": i, "token": token, "id (si connu)": known.get(token, "inconnu")} for i, token in enumerate(tokens)]
    table = pd.DataFrame(rows, columns=["position", "token", "id (si connu)"])
    unknown = sum(token not in known for token in tokens)
    note = (f"**{len(tokens)} tokens** · {unknown} hors vocabulaire · vocabulaire du checkpoint : {len(known):,} tokens."
            if metadata else f"**{len(tokens)} tokens**. Le checkpoint sera utilisé lorsque `checkpoint.pt` sera présent.")
    return note, table, status if status != "ok" else "✅ Modèle prêt"


def _ids_for_prompt(text: str, metadata: dict[str, Any]) -> tuple[list[str], list[int]]:
    tokens = split_tokens(text)
    vocab = metadata["words_to_ids"]
    return tokens, [int(vocab.get(token, 0)) for token in tokens]


@torch.inference_mode()
def inspect_attention(text: str, layer: int, head: int) -> tuple[Any, pd.DataFrame, str]:
    model, metadata, status = load_model()
    if model is None or metadata is None:
        return go.Figure(), pd.DataFrame(), f"⚠️ {status}"
    tokens, ids = _ids_for_prompt(text, metadata)
    if not ids:
        return go.Figure(), pd.DataFrame(), "Écris une phrase pour observer l'attention."
    config = metadata["config"]
    tokens, ids = tokens[: int(config.context_length)], ids[: int(config.context_length)]
    x = torch.tensor([ids], dtype=torch.long, device=metadata["device"])
    x = model.word_token_embedding_table(x)
    positions = torch.arange(x.size(1), device=metadata["device"])
    x = x + model.word_position_embedding_table(positions)
    all_weights: list[torch.Tensor] = []
    all_qkv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for block in model.blocks:
        attn = block.attention
        normed = block.norm_layer_1(x)
        batch, seq, embd = normed.shape
        head_dim = embd // config.num_heads
        q = attn.query_projection(normed).view(batch, seq, config.num_heads, head_dim).transpose(1, 2)
        k = attn.key_projection(normed).view(batch, seq, config.num_heads, head_dim).transpose(1, 2)
        v = attn.value_projection(normed).view(batch, seq, config.num_heads, head_dim).transpose(1, 2)
        weights = q @ k.transpose(-2, -1) / (head_dim**0.5)
        weights = torch.softmax(weights.masked_fill(attn.mask[:seq, :seq] == 0, float("-inf")), dim=-1)
        all_weights.append(weights.detach().cpu())
        all_qkv.append((q.detach().cpu(), k.detach().cpu(), v.detach().cpu()))
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq, embd)
        x = x + attn.output_projection(y)
        x = x + block.mlp(block.norm_layer_2(x))
    layer = max(0, min(int(layer), len(all_weights) - 1))
    head = max(0, min(int(head), int(config.num_heads) - 1))
    matrix = all_weights[layer][0, head, : len(tokens), : len(tokens)].numpy()
    positions = list(range(len(tokens)))
    hover_tokens = np.empty((len(tokens), len(tokens), 2), dtype=object)
    for row, question in enumerate(tokens):
        for column, regarded in enumerate(tokens):
            hover_tokens[row, column] = (question, regarded)
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=positions,
        y=positions,
        customdata=hover_tokens,
        colorscale="Blues",
        zmin=0,
        zmax=1,
        hovertemplate="question : %{customdata[0]}<br>regarde : %{customdata[1]}<br>poids : %{z:.3f}<extra></extra>",
    ))
    axis_tokens = dict(tickmode="array", tickvals=positions, ticktext=tokens)
    fig.update_layout(title=f"Couche {layer + 1} · tête {head + 1} · poids d'attention", xaxis_title="Token regardé (colonne)",
                      yaxis_title="Token qui questionne (ligne)", xaxis=axis_tokens,
                      yaxis={**axis_tokens, "autorange": "reversed"},
                      margin=dict(l=30, r=20, t=60, b=30), height=480, template="plotly_white")
    q, k, v = (value[0, head, : len(tokens)] for value in all_qkv[layer])
    table = pd.DataFrame({"token": tokens, "id": ids, "norme Q": np.linalg.norm(q.cpu().numpy(), axis=1).round(3),
                          "norme K": np.linalg.norm(k.cpu().numpy(), axis=1).round(3), "norme V": np.linalg.norm(v.cpu().numpy(), axis=1).round(3)})
    note = (f"Matrice {len(tokens)} × {len(tokens)}. Chaque ligne est le token qui lit, chaque colonne le token lu. "
            "La zone blanche en haut à droite est le futur masqué : un token ne peut regarder que sa position et celles de gauche. "
            "Q formule la question, K décrit les candidats et V transporte le contenu.")
    return fig, table, note


@torch.inference_mode()
def generate_text(text: str, temperature: float, max_tokens: int) -> tuple[str, str, pd.DataFrame]:
    model, metadata, status = load_model()
    if model is None or metadata is None:
        return "", f"⚠️ {status}", pd.DataFrame()
    tokens, ids = _ids_for_prompt(text, metadata)
    if not ids:
        ids, tokens = [0], [metadata["ids_to_words"].get(0, "<début>")]
    context = torch.tensor([ids], dtype=torch.long, device=metadata["device"])
    generated = model.generate(context, max_new_tokens=int(max_tokens), temperature=max(float(temperature), 0.05))[0].tolist()
    words = [metadata["ids_to_words"].get(int(i), "<inconnu>") for i in generated]
    original_len = len(ids)
    rows = [{"étape": i + 1, "token produit": token} for i, token in enumerate(words[original_len:])]
    return " ".join(words), f"Température **{temperature:.2f}** · appareil **{metadata['device']}**.", pd.DataFrame(rows)


def kv_cache_view(text: str, steps: int) -> tuple[pd.DataFrame, str]:
    """Show the decode-time bookkeeping used by production GPT servers."""
    tokens = split_tokens(text) or ["<prompt>"]
    rows = []
    for step in range(max(1, int(steps))):
        current = tokens[step] if step < len(tokens) else f"token_{step + 1}"
        rows.append({
            "étape": step + 1,
            "token traité": current,
            "Q recalculée": "oui",
            "K réutilisées": max(0, step),
            "V réutilisées": max(0, step),
            "idée": "nouveau token" if step == 0 else "1 Q + cache KV",
        })
    return pd.DataFrame(rows), (
        "**Ce dépôt ne code pas encore ce cache dans `GPT.generate`.** Cette vue montre le mécanisme utilisé par les serveurs optimisés : "
        "pendant le décodage, on calcule la nouvelle question et on réutilise les clés/valeurs des positions précédentes."
    )


CSS = """
:root { --bg:#f7f9fc; --ink:#172033; --muted:#68758a; --accent:#2563eb; --line:#dbe3ef; }
.gradio-container { max-width:1180px !important; background:var(--bg); }
.hero { padding:24px 8px 10px; } .hero h1 { color:var(--ink); margin-bottom:6px; letter-spacing:-.03em; }
.hero p { color:var(--muted); font-size:16px; }
.journey { display:flex; align-items:center; justify-content:center; gap:10px; margin:8px 0 24px; flex-wrap:wrap; }
.journey-step { border:1px solid var(--line); background:white; padding:12px 16px; border-radius:14px; min-width:135px; text-align:center; box-shadow:0 4px 16px rgba(26,48,89,.06); }
.journey-step.active { border-color:#93b4ff; box-shadow:0 7px 22px rgba(37,99,235,.15); }
.journey-step span { display:block; color:var(--accent); font-weight:700; font-size:12px; } .journey-step b { display:block; color:var(--ink); font-size:16px; }
.journey-step small { color:var(--muted); } .journey-arrow { color:#9aa8bc; font-size:24px; } footer { display:none !important; }
.gradio-container .prose, .gradio-container .prose p, .gradio-container .prose li,
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3,
.gradio-container .prose strong { color: var(--ink) !important; }
.gradio-container [data-testid="accordion-content"] .prose,
.gradio-container [data-testid="accordion-content"] .prose p,
.gradio-container [data-testid="accordion-content"] .prose strong { color: #eef2f7 !important; }
.gradio-container button[role="tab"] { color: var(--ink) !important; opacity: 1 !important; }
.gradio-container button[role="tab"][aria-selected="false"] { color: #526178 !important; }
"""


with gr.Blocks(title="GPT expliqué") as demo:
    gr.HTML('<div class="hero"><h1>🧠 GPT, démonté en public</h1><p>Du texte aux tokens, puis de Q/K/V aux probabilités.</p></div>')
    gr.HTML('<div class="journey"><div class="journey-step"><span>1</span><b>Texte</b><small>une phrase</small></div><div class="journey-arrow">→</div><div class="journey-step active"><span>2</span><b>Tokens</b><small>unités manipulées</small></div><div class="journey-arrow">→</div><div class="journey-step"><span>3</span><b>Attention</b><small>Q · K · V</small></div><div class="journey-arrow">→</div><div class="journey-step"><span>4</span><b>Logits</b><small>prochain token</small></div></div>')
    with gr.Tab("1 · Tokens"):
        gr.Markdown("## Le texte n'entre jamais directement dans GPT\nIl est découpé en unités, puis chaque unité devient un identifiant numérique.")
        with gr.Row():
            with gr.Column(scale=1):
                token_input = gr.Textbox(value="Les transformers prédisent le prochain token.", label="Phrase à découper", lines=3)
                token_button = gr.Button("Découper la phrase", variant="primary")
            with gr.Column(scale=1):
                token_note = gr.Markdown(); token_status = gr.Markdown()
        token_output = gr.Dataframe(headers=["position", "token", "id (si connu)"], datatype=["number", "str", "str"], interactive=False)
        token_button.click(tokenise, token_input, [token_note, token_output, token_status]); demo.load(tokenise, token_input, [token_note, token_output, token_status])
    with gr.Tab("2 · Attention Q/K/V"):
        gr.Markdown("## Chaque token choisit ce qui mérite son regard\nL’attention permet à un token de consulter les tokens précédents avant de produire sa représentation.")
        with gr.Accordion("Lire cette vue en 30 secondes", open=True):
            gr.Markdown(
                """
                Imagine un immeuble : une **couche** est un étage de calcul. À chaque étage, le modèle relit la phrase, mais avec une représentation plus travaillée.

                Chaque étage contient plusieurs **têtes d’attention** : ce sont des regards parallèles. Une tête peut privilégier le mot précédent, une autre un sujet ou une relation grammaticale.

                Dans la heatmap, une ligne correspond au token qui pose la question et une colonne au token regardé. **Plus la case est bleue, plus le regard est fort.** La zone blanche en haut à droite correspond au futur masqué : chaque token ne consulte que sa position et les tokens situés à gauche.
                """
            )
        with gr.Row():
            with gr.Column(scale=1):
                attention_input = gr.Textbox(value="le chat mange la souris", label="Phrase courte", lines=2)
                layer = gr.Slider(0, 5, value=0, step=1, label="Couche (étage de calcul)"); head = gr.Slider(0, 11, value=0, step=1, label="Tête (regard spécialisé)")
                gr.Markdown("**Conseil :** garde `le chat mange la souris`, puis change seulement la couche ou la tête et observe les cases de la heatmap.")
                attention_button = gr.Button("Calculer l'attention", variant="primary")
            with gr.Column(scale=2): attention_plot = gr.Plot(label="Attention")
        attention_table = gr.Dataframe(interactive=False, label="Vecteurs réellement calculés"); attention_note = gr.Markdown()
        attention_button.click(inspect_attention, [attention_input, layer, head], [attention_plot, attention_table, attention_note]); demo.load(inspect_attention, [attention_input, layer, head], [attention_plot, attention_table, attention_note])
    with gr.Tab("3 · Génération"):
        gr.Markdown("## GPT propose un token, puis recommence\nLa température règle la part de hasard.")
        with gr.Row():
            with gr.Column(scale=1):
                generation_input = gr.Textbox(value="le chat", label="Début de phrase", lines=2); temperature = gr.Slider(0.2, 1.8, value=0.8, step=0.1, label="Température"); max_tokens = gr.Slider(1, 30, value=12, step=1, label="Nombre de tokens")
                generation_button = gr.Button("Générer", variant="primary")
            with gr.Column(scale=2): generation_output = gr.Textbox(label="Séquence complète", lines=5, interactive=False); generation_note = gr.Markdown()
        generation_steps = gr.Dataframe(interactive=False, label="Pas de génération")
        generation_button.click(generate_text, [generation_input, temperature, max_tokens], [generation_output, generation_note, generation_steps])
    with gr.Tab("4 · KV cache"):
        gr.Markdown("## Pourquoi la génération peut être accélérée\nLe cache conserve les clés et valeurs déjà calculées. À chaque nouveau token, seule la nouvelle question doit être calculée.")
        with gr.Row():
            with gr.Column(scale=1):
                cache_input = gr.Textbox(value="le chat mange", label="Contexte", lines=2)
                cache_steps = gr.Slider(1, 12, value=6, step=1, label="Nombre d'étapes")
                cache_button = gr.Button("Dérouler le cache", variant="primary")
            with gr.Column(scale=2):
                cache_note = gr.Markdown()
        cache_table = gr.Dataframe(interactive=False, label="Ce qui est recalculé et réutilisé")
        cache_button.click(kv_cache_view, [cache_input, cache_steps], [cache_table, cache_note])
        demo.load(kv_cache_view, [cache_input, cache_steps], [cache_table, cache_note])
    with gr.Tab("5 · Comprendre"):
        gr.Markdown("""
        ## Les notions visibles dans cette démo

        - **Token** : une unité du vocabulaire, représentée par un entier.
        - **Embedding** : un vecteur appris qui donne une position au token.
        - **Transformer** : des blocs qui mélangent attention et réseau feed-forward.
        - **Q/K/V** : question, clé et contenu ; Q·K mesure l'affinité avant de pondérer V.
        - **Logits** : scores produits pour tous les tokens possibles avant la softmax.
        - **KV cache** : clés et valeurs conservées pendant la génération pour éviter de refaire le calcul.

        Le modèle de ce dépôt est volontairement petit et pédagogique. Il rend le mécanisme observable, mais ses sorties ne sont pas celles d'un modèle généraliste moderne.
        """)


if __name__ == "__main__":
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    )
