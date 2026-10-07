"""
Training con fine-tuning parziale di MobileNetV2.

Strategia:
  - backbone congelato tranne gli ultimi blocchi (features[15:]), che vengono sbloccati
    con learning rate basso (ritocco leggero di ciò che ImageNet ha già insegnato);
  - layer finale nuovo, con learning rate più alto (parte da pesi casuali).
Con il solo layer finale allenabile la val acc si fermava al ~74%; con questo
fine-tuning arriva al ~97%.

ATTENZIONE: rilanciare questo script sovrascrive il checkpoint in models/ (il
salvataggio parte da best_val_acc = 0). Se vuoi tenere quello attuale, copialo prima.
"""

import torch
import torch.nn as nn

from config import BEST_MODEL_PATH, MODELS_DIR
from dataset import train_loader, val_loader
from model import build_model

NUM_EPOCHS = 5
LR_BACKBONE = 1e-4   # basso: ritocchi leggeri ai blocchi sbloccati
LR_HEAD = 1e-3       # più alto: il layer finale parte da zero


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    # --- Fine-tuning parziale ---
    # Sblocca gli ultimi blocchi del backbone (features.15 ... features.18): sono quelli
    # che combinano le feature in concetti di alto livello. Gli strati iniziali (bordi,
    # texture) restano congelati perché sono generici.
    # Va fatto PRIMA di creare l'optimizer, che legge requires_grad a quel momento.
    for param in model.features[15:].parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.features[15:].parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(),    "lr": LR_HEAD},
    ])

    MODELS_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # ---------- TRAINING ----------
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()              # azzera i gradienti del batch precedente
            outputs = model(images)            # forward pass
            loss = criterion(outputs, labels)  # calcolo della loss
            loss.backward()                    # backward pass: calcola i gradienti
            optimizer.step()                   # aggiorna i pesi allenabili

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---------- VALIDATION ----------
        model.eval()          # disattiva Dropout: la valutazione deve essere deterministica
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # nessun gradiente: qui non aggiorniamo nulla
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                predicted = outputs.argmax(dim=1)   # classe con il punteggio più alto
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}")

        # ---------- CHECKPOINT ----------
        # Salva solo se questa epoca è la migliore finora sulla validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> nuovo best model salvato (val acc {val_acc:.4f})")


if __name__ == "__main__":
    main()
