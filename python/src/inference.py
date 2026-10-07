"""
Inferenza: carica il modello allenato e predice il gesto da un tensore già preprocessato.

Il tensore atteso (3, 224, 224, float32) deve essere preprocessato ESATTAMENTE come in
training: resize 224x224, valori in [0, 1], ordine canali RGB, poi normalizzazione con
mean/std di ImageNet. Quando collegheremo la pipeline CUDA, dovrà produrre questo.

Test rapido da riga di comando:
    python3 inference.py percorso/immagine.jpg
"""

import torch

from config import BEST_MODEL_PATH, CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import build_model


def load_model(checkpoint_path=BEST_MODEL_PATH):
    """Ricostruisce l'architettura e carica i pesi allenati. Restituisce (modello, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)   # i pesi ImageNet non servono: li sovrascrive il checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()                            # Dropout spento: inferenza deterministica
    return model, device


def predict(model, device, tensor):
    """
    tensor: float32 di forma (3, 224, 224), già normalizzato con mean/std ImageNet.
    Restituisce (nome_gesto, confidenza tra 0 e 1).
    """
    with torch.no_grad():
        # unsqueeze(0) aggiunge la dimensione batch: la rete vuole (N, 3, 224, 224).
        logits = model(tensor.unsqueeze(0).to(device))
        # Per l'argmax la softmax non servirebbe (non cambia l'ordine dei punteggi);
        # la usiamo perché ci interessa la confidenza come probabilità.
        probs = torch.softmax(logits, dim=1)[0]
    confidence, idx = probs.max(dim=0)
    return CLASSES[idx.item()], confidence.item()


if __name__ == "__main__":
    import sys
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    model, device = load_model()
    image = Image.open(sys.argv[1]).convert("RGB")
    gesture, confidence = predict(model, device, transform(image))
    print(f"Gesto: {gesture} (confidenza {confidence:.2%})")
