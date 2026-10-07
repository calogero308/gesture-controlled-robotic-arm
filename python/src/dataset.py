import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import DATASET_ROOT, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# --- Parametri ---
BATCH_SIZE = 32
TRAIN_FRACTION = 0.8   # scelta arbitraria (non un vincolo tecnico): 80% train / 20% validation
SPLIT_SEED = 42        # stesso seed = stesso split a ogni esecuzione (esperimenti confrontabili)


# --- Trasformazioni ---
# Applicate immagine per immagine, al momento del caricamento (dentro __getitem__
# di ImageFolder), non tutte insieme all'inizio.
#
# Training: include data augmentation (rotazioni leggere + variazioni di
# luminosità/contrasto), assente invece in validation: la validation deve restare
# "pulita" per misurare le prestazioni reali del modello su dati non alterati.
# Niente RandomHorizontalFlip: lo scartiamo di proposito per non rischiare di
# alterare gesti che dipendono dall'orientamento.
#
# L'augmentation lavora su immagini PIL, quindi va PRIMA di ToTensor().

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),            # MobileNetV2 si aspetta 224x224
    transforms.RandomRotation(degrees=15),                  # ruota casualmente tra -15 e +15 gradi
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # varia luminosità/contrasto del ±20%
    transforms.ToTensor(),                                  # PIL -> tensore (C, H, W), valori in [0, 1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation: stesso preprocessing del training (dimensione, normalizzazione), ma
# senza alcuna augmentation.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# --- Caricamento dataset ---
# ImageFolder legge una struttura "una cartella per classe" (fist/, like/, palm/,
# peace/, rock/) e costruisce da solo l'elenco (percorso_immagine, indice_classe).
#
# Due istanze separate (full_train, full_val) sulla stessa cartella perché ImageFolder
# applica UNA sola trasformazione a tutto il dataset: per avere transform diverse su
# train e validation servono due oggetti distinti, poi "tagliati" con gli stessi
# indici per restare coerenti.

full_train = ImageFolder(root=DATASET_ROOT, transform=train_transform)
full_val   = ImageFolder(root=DATASET_ROOT, transform=val_transform)


# --- Split train/validation ---
train_size = int(TRAIN_FRACTION * len(full_train))
val_size   = len(full_train) - train_size

# manual_seed fissa la sequenza "casuale": stesso split ogni volta che il file
# viene eseguito.
generator = torch.Generator().manual_seed(SPLIT_SEED)

# random_split lavora sugli INDICI (range(len(...))), non sulle immagini: produce due
# liste di indici disgiunte.
train_indices, val_indices = random_split(range(len(full_train)), [train_size, val_size], generator=generator)

# Subset applica quegli indici al dataset "giusto" (quello con la transform corretta):
# stesse immagini fisiche, ma ogni subset le carica con la propria pipeline.
train_dataset = Subset(full_train, train_indices)
val_dataset   = Subset(full_val, val_indices)


# --- DataLoader ---
# Raggruppano il dataset in batch e si occupano dello shuffle.
# shuffle=True in training: mescola l'ordine a ogni epoca, per non imparare pattern
# legati all'ordine delle immagini.
# shuffle=False in validation: l'ordine non cambia la metrica, e tenerlo fisso rende
# i risultati riproducibili e ispezionabili.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
