import numpy as np
import ctypes
from pathlib import Path
from config import CUDA_LIB_PATH

# Carico la libreria che farà interfacciare python con CUDA
lib = ctypes.CDLL(str(CUDA_LIB_PATH))

# Dichiaro i tipi dei parametri di Process_image, così ctypes sa come impacchettarli
# L'ordine deve essere lo stesso della firma definita nel kernel di partenza
lib.process_image.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),     # h_input
    ctypes.POINTER(ctypes.c_float),     # h_output
    ctypes.c_int,                       # srcWidth
    ctypes.c_int,                       # srcHeight
    ctypes.c_int,                       # channels
    ctypes.c_int,                       # dstWidth
    ctypes.c_int                        # dstHeight
]

# Imposto il tipo di ritorno che avrà la funzione
lib.process_image.restype = None

# Funzione che effettuerà il preprocessing
def preprocess(image_rgb, size=224):

    # Effettuo il controllo sul tipo di immagine ricevuto e sulle dimensioni in modo da 
    # individuare possibili errori silenziosi che potrebbero rompere il codice
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"Atteso uint8, ricevuto {image_rgb.dtype}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Atteso array (H, W, 3), ricevuto shape {image_rgb.shape}")

    # Garantisce che i byte siano contigui in memoria (riga dopo riga).
    # Se l'array lo è già, non fa nulla; altrimenti ne crea una copia contigua.
    # Riassegnando alla stessa variabile, input_ptr più sotto punta sempre ai dati giusti.
    image_rgb = np.ascontiguousarray(image_rgb)

    # Ottengo le dimensioni di partenza
    srcHeight, srcWidth, channels = image_rgb.shape

    # Dimensioni di destinazione
    dstWidth, dstHeight = size, size

    # Preparo il buffer di output
    output = np.zeros((channels, dstHeight, dstWidth), dtype = np.float32)  

    # Conversione dell'immagine in un puntatore da poter passare alla funzione
    # .ctypes.data_as(...) prende l'indirizzo di memoria reale dove NumPy tiene i dati dell'array
    input_ptr = image_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Chiamata alla funzione effettiva
    lib.process_image(input_ptr, output_ptr, srcWidth, srcHeight, channels, dstWidth, dstHeight)

    return output