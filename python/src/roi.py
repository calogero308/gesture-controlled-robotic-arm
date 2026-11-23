import cv2

# Funzione che calcola il box dove verrà inserito il gesto
def roi_box(height, width, fraction = 0.6):
    
    # Calcolo del lato minimo
    shortest_l = min(height, width)

    # Calcolo della dimensione del quadrato di riconoscimento
    square_l = int(fraction * shortest_l)

    # Calcolo dei punti di partenza
    y0 = (height - square_l) // 2
    x0 = (width - square_l) // 2

    return y0, x0, square_l

# Ritaglia il box per passarlo al modello
def crop_roi(frame, fraction = 0.6):

    # Inizializzazione delle variabili delle dimensioni
    height, width = frame.shape[:2]

    # Chiamata alla funzione per il calcolo del quadrato
    y0, x0, side = roi_box(height, width, fraction)

    # Ritaglio dell'immagine
    cropped_img = frame[y0 : y0 + side, x0 : x0 + side]

    return cropped_img
