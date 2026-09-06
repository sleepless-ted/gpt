"""Interactive Gradio tour of the small GPT model in this repository."""

from __future__ import annotations

import os
import re
import time
from html import escape
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


def _compact_number(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} Md"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"
    if value >= 1_000:
        return f"{value / 1_000:.1f} k"
    return str(value)


def _architecture_values(
    text: str,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    context_length: int,
) -> dict[str, Any]:
    tokens = split_tokens(text)[: int(context_length)] or ["<vide>"]
    layers = max(1, int(num_layers))
    heads = max(1, int(num_heads))
    dimension = max(8, int(head_dim))
    context = max(1, int(context_length))
    embedding_dim = heads * dimension
    _, metadata, _ = load_model()
    vocab_size = int(getattr(metadata["config"], "vocab_size", 50_000)) if metadata else 50_000
    # Embeddings + positions + Transformer blocks + final LayerNorm.
    # The output projection shares the token embedding weights in this repo.
    block_parameters = layers * (12 * embedding_dim * embedding_dim + 9 * embedding_dim)
    estimated_parameters = vocab_size * embedding_dim + context * embedding_dim + block_parameters + 2 * embedding_dim
    return {
        "tokens": tokens,
        "layers": layers,
        "heads": heads,
        "head_dim": dimension,
        "context": context,
        "embedding_dim": embedding_dim,
        "vocab_size": vocab_size,
        "estimated_parameters": estimated_parameters,
    }


def architecture_view(
    text: str,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    context_length: int,
) -> str:
    values = _architecture_values(text, num_layers, num_heads, head_dim, context_length)
    tokens = values["tokens"]
    shown_tokens = tokens[:12]
    token_html = "".join(f'<span class="arch-token">{escape(token)}</span>' for token in shown_tokens)
    if len(tokens) > len(shown_tokens):
        token_html += f'<span class="arch-token muted">+{len(tokens) - len(shown_tokens)}</span>'

    head_count = min(values["heads"], 16)
    head_dots = "".join('<span class="head-dot"></span>' for _ in range(head_count))
    if values["heads"] > head_count:
        head_dots += f'<span class="head-more">+{values["heads"] - head_count}</span>'

    layer_cards = []
    for index in range(values["layers"]):
        layer_cards.append(
            '<div class="layer-card">'
            f'<div class="layer-title">Bloc {index + 1}</div>'
            f'<div class="head-dots" aria-label="{values["heads"]} têtes">{head_dots}</div>'
            '<div class="layer-ops"><span>Attention</span><span>MLP ×4</span></div>'
            '</div>'
        )

    n_tokens = len(tokens)
    active_attention = n_tokens * (n_tokens + 1) // 2
    total_attention = n_tokens * n_tokens
    return f"""
    <div class="architecture" aria-label="Schéma du Transformer simulé">
      <div class="architecture-stats">
        <div><span>Tokens</span><strong>{n_tokens}</strong></div>
        <div><span>d_model</span><strong>{values['embedding_dim']}</strong></div>
        <div><span>d_head</span><strong>{values['head_dim']}</strong></div>
        <div><span>Paramètres estimés</span><strong>{_compact_number(values['estimated_parameters'])}</strong></div>
      </div>
      <div class="architecture-flow">
        <div class="flow-node token-node"><b>Tokens</b><div class="arch-tokens">{token_html}</div></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node"><b>Embeddings</b><small>{n_tokens} × {values['embedding_dim']}</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node active-node"><b>{values['layers']} blocs Transformer</b><small>{values['heads']} têtes × {values['head_dim']} dimensions</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node"><b>Logits</b><small>{values['vocab_size']:,} scores</small></div>
      </div>
      <div class="layer-grid">{''.join(layer_cards)}</div>
      <div class="causal-summary">
        Attention causale : <strong>{n_tokens} × {n_tokens}</strong> cases par tête,
        dont <strong>{active_attention}/{total_attention}</strong> accessibles.
        Contexte maximal choisi : <strong>{values['context']}</strong> tokens.
      </div>
    </div>
    """


def process_animation(
    text: str,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    context_length: int,
) -> str:
    values = _architecture_values(text, num_layers, num_heads, head_dim, context_length)
    shown_tokens = values["tokens"][:10]
    run_id = time.monotonic_ns()

    def animated_tokens(stage: int, suffix: str = "") -> str:
        return "".join(
            f'<span class="process-token" style="--token:{index};--stage:{stage}">{escape(token)}{suffix}</span>'
            for index, token in enumerate(shown_tokens)
        )

    stages = [
        ("1", "Tokenisation", animated_tokens(0), f"{len(values['tokens'])} unités"),
        ("2", "Embeddings", animated_tokens(1, "⃗"), f"vecteurs de {values['embedding_dim']} nombres"),
        ("3", "Attention Q/K/V", animated_tokens(2), f"{values['heads']} regards parallèles"),
        ("4", "Blocs Transformer", animated_tokens(3), f"répété {values['layers']} fois"),
        ("5", "Logits", '<span class="process-token prediction" style="--token:0;--stage:4">prochain token ?</span>', f"{values['vocab_size']:,} possibilités"),
    ]
    stage_html = "".join(
        f"""
        <div class="process-stage" style="--stage:{index}">
          <div class="stage-number">{number}</div>
          <div class="stage-copy"><b>{title}</b><small>{detail}</small></div>
          <div class="stage-token-line">{content}</div>
        </div>
        {('<div class="process-connector" style="--stage:' + str(index) + '"><span>↓</span></div>') if index < len(stages) - 1 else ''}
        """
        for index, (number, title, content, detail) in enumerate(stages)
    )
    return f"""
    <div class="process-animation" data-run="{run_id}" aria-label="Animation du traitement de la phrase">
      <div class="animation-title"><b>La phrase traverse le modèle</b><span>L’animation se joue une fois à chaque clic.</span></div>
      {stage_html}
    </div>
    """


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
.architecture { color:var(--ink); padding:4px 2px 12px; }
.architecture-stats { display:grid; grid-template-columns:repeat(4,minmax(110px,1fr)); gap:10px; margin-bottom:16px; }
.architecture-stats div { background:white; border:1px solid var(--line); border-radius:12px; padding:11px 13px; }
.architecture-stats span { color:var(--muted); display:block; font-size:12px; }
.architecture-stats strong { display:block; font-size:20px; margin-top:2px; }
.architecture-flow { display:flex; align-items:stretch; gap:8px; margin:4px 0 16px; }
.flow-node { background:white; border:1px solid var(--line); border-radius:13px; padding:12px; min-height:72px; flex:1; display:flex; flex-direction:column; justify-content:center; }
.flow-node.active-node { border-color:#82aaff; background:#edf4ff; }
.flow-node b { color:var(--ink) !important; }
.flow-node small { color:var(--muted); display:block; margin-top:4px; }
.flow-arrow { align-self:center; color:#8fa0b8; font-size:22px; }
.token-node { flex:1.35; }
.arch-tokens { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.arch-token { background:#e8efff; color:#19459b; border-radius:6px; padding:2px 6px; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }
.arch-token.muted { color:var(--muted); background:#edf0f5; }
.layer-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(125px,1fr)); gap:8px; }
.layer-card { background:#172033; color:#eef4ff; border-radius:11px; padding:10px; min-height:82px; }
.layer-title { color:#eef4ff; font-weight:700; margin-bottom:7px; }
.head-dots { display:flex; flex-wrap:wrap; gap:4px; min-height:15px; align-items:center; }
.head-dot { width:7px; height:7px; background:#73a4ff; border-radius:50%; box-shadow:0 0 7px rgba(115,164,255,.65); }
.head-more { font-size:10px; color:#b9cdf7; }
.layer-ops { display:flex; gap:5px; margin-top:8px; }
.layer-ops span { background:#29364d; border-radius:5px; padding:3px 5px; font-size:10px; }
.causal-summary { color:var(--muted); margin-top:12px; }
.causal-summary strong { color:var(--ink); }
.process-animation { color:var(--ink); margin-top:14px; padding:16px; background:white; border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.animation-title { display:flex; justify-content:space-between; gap:12px; margin-bottom:12px; }
.animation-title b,.stage-copy b { color:var(--ink) !important; }
.animation-title span { color:var(--muted); font-size:12px; }
.process-stage { display:grid; grid-template-columns:34px 170px 1fr; align-items:center; gap:10px; min-height:60px; animation:stage-pass .8s ease-out both; animation-delay:calc(var(--stage) * .9s); }
.stage-number { width:30px; height:30px; display:grid; place-items:center; border-radius:50%; background:#e8efff; color:#1d55c6; font-weight:700; }
.stage-copy b,.stage-copy small { display:block; }
.stage-copy small { color:var(--muted); margin-top:2px; }
.stage-token-line { display:flex; flex-wrap:wrap; gap:5px; min-height:28px; align-items:center; }
.process-token { background:#2563eb; color:white; border-radius:7px; padding:4px 7px; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; opacity:0; transform:translateX(-14px) scale(.92); animation:token-arrive .48s ease-out forwards; animation-delay:calc(var(--stage) * .9s + var(--token) * .07s + .12s); }
.process-token.prediction { background:#7c3aed; }
.process-connector { height:15px; margin-left:14px; color:#7a8ba4; opacity:0; animation:connector-pass .35s ease-out forwards; animation-delay:calc(var(--stage) * .9s + .7s); }
@keyframes stage-pass {
  from { transform:translateX(-8px); background:rgba(37,99,235,.07); }
  to { transform:translateX(0); background:transparent; }
}
@keyframes token-arrive { to { opacity:1; transform:translateX(0) scale(1); box-shadow:0 4px 12px rgba(37,99,235,.22); } }
@keyframes connector-pass { to { opacity:1; transform:translateY(3px); } }
@media (max-width:760px) {
  .architecture-stats { grid-template-columns:repeat(2,minmax(100px,1fr)); }
  .architecture-flow { flex-direction:column; }
  .flow-arrow { transform:rotate(90deg); }
  .process-stage { grid-template-columns:34px 1fr; }
  .stage-token-line { grid-column:2; }
}
@media (prefers-reduced-motion:reduce) {
  .process-stage,.process-token,.process-connector { animation:none; opacity:1; transform:none; }
}
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
    with gr.Tab("5 · Architecture animée"):
        gr.Markdown(
            "## Construis un Transformer et regarde la phrase le traverser\n"
            "Les réglages modifient un **schéma simulé**. Les valeurs par défaut correspondent au petit GPT de ce dépôt."
        )
        with gr.Row():
            with gr.Column(scale=1):
                architecture_input = gr.Textbox(value="le chat mange la souris", label="Phrase traitée", lines=2)
                architecture_layers = gr.Slider(1, 12, value=6, step=1, label="Nombre de couches")
                architecture_heads = gr.Slider(1, 16, value=12, step=1, label="Têtes par couche")
                architecture_head_dim = gr.Slider(8, 128, value=32, step=8, label="Dimension d’une tête")
                architecture_context = gr.Slider(8, 128, value=32, step=8, label="Contexte maximal")
                animation_button = gr.Button("▶ Relancer le parcours", variant="primary")
            with gr.Column(scale=2):
                architecture_html = gr.HTML()
        animation_html = gr.HTML()
        architecture_controls = [architecture_input, architecture_layers, architecture_heads, architecture_head_dim, architecture_context]
        for architecture_control in architecture_controls:
            architecture_control.change(architecture_view, architecture_controls, architecture_html, queue=False)
        animation_button.click(process_animation, architecture_controls, animation_html, queue=False)
        demo.load(architecture_view, architecture_controls, architecture_html)
        demo.load(process_animation, architecture_controls, animation_html)
    with gr.Tab("6 · Comprendre"):
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
