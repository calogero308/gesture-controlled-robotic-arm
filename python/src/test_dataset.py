"""
Controlli di sanità sul dataset: percorso, classi, split, forma dei dati, augmentation.

Lancialo ogni volta che tocchi dataset.py, config.py o la cartella dei dati.
"""

import torch
import matplotlib
matplotlib.use("Agg")   # backend senza finestra: salva solo su file (da WSL non serve un display)
import matplotlib.pyplot as plt

from config import CLASSES, DATA_DIR, IMAGENET_MEAN, IMAGENET_STD
from dataset import full_train, train_dataset, val_dataset, train_loader, train_indices, val_indices

# --- Classi e conteggi ---
print("Classi trovate:", full_train.classes)
assert full_train.classes == CLASSES, (
    "Le classi trovate non coincidono con config.CLASSES: c'è una cartella di troppo "
    "(o mancante) dentro DATASET_ROOT?"
)
print("Totale immagini:", len(full_train))
print("Immagini in training:", len(train_dataset))
print("Immagini in validation:", len(val_dataset))

# --- Train e validation non devono condividere immagini ---
overlap = set(train_indices.indices) & set(val_indices.indices)
assert len(overlap) == 0, f"{len(overlap)} immagini compaiono sia in train che in validation"
print("Nessuna sovrapposizione tra train e validation")

# --- Forma di un elemento e di un batch ---
image, label = train_dataset[0]
print("Shape immagine:", image.shape)
print("Etichetta:", label)

images, labels = next(iter(train_loader))
print("Shape batch:", images.shape, "- etichette:", labels.shape)

# --- Verifica visiva dell'augmentation ---
# Stessa immagine (indice 0) richiamata più volte: ogni chiamata riapplica la
# trasformazione casuale, quindi le 4 varianti sono diverse tra loro.
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax in axes:
    image, label = full_train[0]
    image = image * std + mean                    # inverte la Normalize per poterla vedere
    image = image.permute(1, 2, 0).clamp(0, 1)    # da (C, H, W) a (H, W, C) per matplotlib
    ax.imshow(image)
    ax.set_title(full_train.classes[label])
    ax.axis("off")

output_path = DATA_DIR / "augmentation_check.png"
plt.savefig(output_path)
print("Salvato in", output_path)
