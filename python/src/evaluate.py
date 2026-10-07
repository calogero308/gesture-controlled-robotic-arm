"""
Valutazione del modello salvato sul validation set: matrice di confusione e
accuratezza per classe.

Importare dataset.py ricrea lo stesso split (seed fisso): val_loader contiene
esattamente le stesse immagini usate in validazione durante il training.
"""

import torch

from config import BEST_MODEL_PATH
from dataset import val_loader, full_val
from model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pretrained=False: non serve scaricare i pesi ImageNet, li sovrascrive il checkpoint.
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    classes = full_val.classes
    num_classes = len(classes)

    # Matrice di confusione: riga = classe vera, colonna = classe predetta.
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            predicted = model(images).argmax(dim=1).cpu()
            for true, pred in zip(labels, predicted):
                confusion[true, pred] += 1

    print("Matrice di confusione (righe = vero, colonne = predetto):")
    print("classi:", classes)
    print(confusion)

    print("\nAccuratezza per classe:")
    for i, name in enumerate(classes):
        class_acc = confusion[i, i].item() / confusion[i].sum().item()
        print(f"  {name}: {class_acc:.4f}")

    overall = confusion.diag().sum().item() / confusion.sum().item()
    print(f"\nAccuratezza totale: {overall:.4f}")


if __name__ == "__main__":
    main()
