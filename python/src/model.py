"""
Definizione dell'architettura: l'UNICO posto in cui è descritto il modello.

train.py, evaluate.py e inference.py importano build_model() da qui, così
l'architettura non può "andare fuori sincronia" tra training e inferenza (altrimenti
i pesi salvati non si caricherebbero, o si caricherebbero su una struttura diversa).
"""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    MobileNetV2 con backbone congelato e un nuovo layer finale per `num_classes` classi.

    pretrained=True  -> parte dai pesi ImageNet (serve in training).
    pretrained=False -> struttura con pesi casuali, senza scaricare nulla (basta in
                        valutazione/inferenza, dove i pesi li sovrascrive il checkpoint).

    Lo sblocco degli ultimi blocchi per il fine-tuning NON è qui: è una scelta di
    training e sta in train.py.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Congela tutto il backbone: i suoi pesi non vengono aggiornati dall'optimizer.
    for param in model.parameters():
        param.requires_grad = False

    # Sostituisce l'ultimo layer (1280 -> 1000 classi ImageNet) con uno nuovo.
    # 1280 è fisso (dipende dall'architettura), num_classes dipende dal nostro problema.
    # Un layer appena creato nasce con requires_grad=True: sarà allenabile.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    return model
