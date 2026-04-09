from datasets import load_dataset
from tqdm import tqdm
import os

print("Téléchargement d'un échantillon de FineWeb-Edu ...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

with open("fineweb_edu.txt", "w", encoding="utf-8") as f:
    for i, example in tqdm(enumerate(dataset), total=200000):
        if i >= 200000: break                    # ~ 500-800 Mo de texte (suffisant pour commencer)
        text = example["text"].strip()
        if len(text) > 100:                      # on garde seulement les textes corrects
            f.write(text + "\n\n")

print("✅ Fichier créé : fineweb_edu.txt")
print("Taille :", os.path.getsize("fineweb_edu.txt") / (1024*1024), "Mo")