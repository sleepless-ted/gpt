---
title: GPT expliqué
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.11"
---

# GPT expliqué

Une interface Gradio pour découvrir le petit GPT de ce dépôt : tokenisation,
embeddings, attention causale Q/K/V et génération token par token.

## Lancer localement avec Pixi

```powershell
pixi install
pixi run app
```

Pour vérifier la syntaxe sans démarrer l'interface :

```powershell
pixi run check
```

Le fichier `checkpoint.pt` est nécessaire pour les vues d'attention et de
génération. La vue de tokenisation reste disponible sans lui. Comme le
checkpoint est volumineux, tu peux le publier dans un dépôt **Model** privé ou
public, puis créer la variable `MODEL_REPO` dans les Settings du Space (par
exemple `sleepless-ted/gpt-checkpoint`). `HF_TOKEN` est uniquement nécessaire
pour un dépôt privé.
