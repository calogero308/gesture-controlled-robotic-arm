"""

Scarica il dataset HaGRID (classificazione, 512 px) da Hugging Face e prepara la
struttura che si aspetta config.py:

    data/hagrid/hagrid-classification-512p-no-gesture-150k/<classe>/<immagini>

Passi: download dell'archivio zip (se non è già in data/downloads/) -> estrazione delle
sole classi in config.CLASSES, rispettando il tetto per classe di
config.MAX_IMAGES_PER_CLASS (es. no_gesture) -> conteggio per classe.

Richiede:  pip install huggingface_hub
Uso (da qualsiasi cartella):
    python3 python/scripts/download_dataset.py

Note:
  - l'archivio pesa diversi GB: servono spazio su disco e tempo;
  - lo zip scaricato resta in data/downloads/ ; a estrazione
    finita si può cancellare a mano per liberare spazio;
  - se tutte le classi di config.CLASSES sono già presenti in DATASET_ROOT lo script
    non fa nulla; se uno zip è già in data/downloads/ non lo riscarica;
  - se un'estrazione viene interrotta, rilanciandolo salta i file già estratti;
  - con un tetto per classe si prendono i primi file che compaiono nello zip: la
    selezione è deterministica, non casuale.

"""

import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

# config.py sta in python/src: lo rendiamo importabile da qui
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CLASSES, DATA_DIR, DATASET_ROOT, MAX_IMAGES_PER_CLASS

REPO_ID = "cj-mills/hagrid-classification-512p-no-gesture-150k-zip"
DOWNLOAD_DIR = DATA_DIR / "downloads"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def count_images():
    """Numero di immagini già presenti per ciascuna classe in DATASET_ROOT."""
    counts = {}
    for name in CLASSES:
        folder = DATASET_ROOT / name
        counts[name] = (
            sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            if folder.is_dir() else 0
        )
    return counts


def download():
    """Scarica il repository in data/downloads/ e restituisce i percorsi degli zip trovati."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=DOWNLOAD_DIR)
    zips = sorted(DOWNLOAD_DIR.rglob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"Nessun file .zip trovato in {DOWNLOAD_DIR}")
    return zips


def extract_classes(zip_path):
    """
    Estrae dallo zip solo le immagini il cui cartella-genitore è una delle classi volute,
    e le copia in DATASET_ROOT/<classe>/<file>. Funziona a qualunque profondità
    dell'archivio, perché guarda solo la cartella che contiene direttamente il file.
    Se una classe ha un tetto in MAX_IMAGES_PER_CLASS, smette di estrarla quando la sua
    cartella (file già presenti + nuovi) lo raggiunge.
    """
    wanted = set(CLASSES)
    counts = count_images()      # immagini già su disco per classe, aggiornato mentre estraiamo
    extracted = 0
    already_there = 0
    over_cap = 0

    with zipfile.ZipFile(zip_path) as archive:
        # Itera dentro l'archivio
        for member in archive.infolist():

            # Controlla che sia una cartella
            if member.is_dir():
                continue
            path = Path(member.filename)

            # Controlla la presenza in IMAGE_SUFFIXES
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            class_name = path.parent.name

            # Controlla se viene richiesta questa determinata classe
            if class_name not in wanted:
                continue

            # Crea il target dove inserire il file
            target = DATASET_ROOT / class_name / path.name
            if target.exists():
                already_there += 1   # estrazione ripresa, o nome file ripetuto in sottocartelle diverse
                continue             # (già incluso in counts, non si incrementa)

            cap = MAX_IMAGES_PER_CLASS.get(class_name)
            if cap is not None and counts[class_name] >= cap:
                over_cap += 1        # tetto raggiunto per questa classe
                continue

            # Creazione della directory e inserimento del file
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1
            counts[class_name] += 1

    print(f"  estratte {extracted} immagini ({already_there} già presenti, "
          f"{over_cap} saltate per il tetto per classe)")


def main():
    counts = count_images()
    if all(n > 0 for n in counts.values()):
        print("Dataset già presente, niente da fare:")
    else:
        zips = sorted(DOWNLOAD_DIR.rglob("*.zip"))
        if zips:
            print(f"Zip già presente in {DOWNLOAD_DIR}, salto il download.")
        else:
            print(f"Download da Hugging Face ({REPO_ID}) in {DOWNLOAD_DIR} ...")
            zips = download()
        for zip_path in zips:
            print(f"Estrazione di {zip_path.name} in {DATASET_ROOT} ...")
            extract_classes(zip_path)
        counts = count_images()
        print("\nEstrazione terminata:")

    for name, n in counts.items():
        print(f"  {name}: {n}")
    print(f"  totale: {sum(counts.values())}")

    # ImageFolder trasforma ogni sottocartella in una classe: una cartella di troppo
    # darebbe una classe in più e farebbe fallire gli assert di test_dataset.py.
    extra = [d.name for d in DATASET_ROOT.iterdir() if d.is_dir() and d.name not in CLASSES]
    if extra:
        print(f"\nATTENZIONE: cartelle extra in {DATASET_ROOT}: {extra} (verrebbero lette come classi)")
    if any(n == 0 for n in counts.values()):
        print("\nATTENZIONE: almeno una classe è vuota: controlla la struttura dello zip.")


if __name__ == "__main__":
    main()