"""
Configurazione centrale del progetto: percorsi e costanti condivise.

Perché esiste: percorsi, nomi delle classi e valori di normalizzazione servono a più
file (dataset, training, valutazione, inferenza, e in futuro l'integrazione CUDA).
Definirli qui, una volta sola, evita di ripeterli (e di dimenticarne uno quando cambia
qualcosa).
"""

from pathlib import Path

# --- Percorsi ---
# __file__ = questo file (python/src/config.py), risolto in modo affidabile
# indipendentemente da dove viene lanciato lo script.
# .parent (src) -> .parent (python) -> .parent (radice del progetto)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# Cartella letta da ImageFolder: deve contenere SOLO le sottocartelle delle classi
# (fist/, like/, ...). Qualunque altra cartella messa qui dentro verrebbe contata
# come una classe in più e romperebbe il training.
DATASET_ROOT = DATA_DIR / "hagrid" / "hagrid-classification-512p-no-gesture-150k"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model_finetuned.pth"

# --- Classi ---
# Ordine alfabetico = ordine con cui ImageFolder assegna gli indici 0..4.
CLASSES = ["fist", "like", "palm", "peace", "rock"]
NUM_CLASSES = len(CLASSES)

# --- Preprocessing ---
# MobileNetV2 (pretrained su ImageNet) si aspetta input 224x224 normalizzati con
# la media/deviazione standard di ImageNet: NON sono valori a piacere, sono quelli
# con cui il backbone è stato allenato. Training e inferenza devono usare gli stessi.
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
