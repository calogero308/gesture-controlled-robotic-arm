import cv2
import numpy as np
import ctypes

# Carico l'immagine di test
img = cv2.imread('test.jpeg')

# Ottengo le dimensioni di partenza
srcHeight, srcWidth, channels = img.shape   # altezza prima, larghezza dopo

# Definisco le dimensioni di destinazione
dstWidth = 224
dstHeight = 224

# Preparo il buffer di output
output = np.zeros((channels, dstHeight, dstWidth), dtype = np.float32) # uint8 equivalente di unsigned char

print(output.shape)

# Carico la libreria condivisa
lib = ctypes.CDLL('./cuda_wrappers.so')

# Dichiaro i tipi dei parametri di run_resize, così ctypes sa come impacchettarli
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

# Conversione dell'immagine in un puntatore da poter passare alla funzione
# .ctypes.data_as(...) prende l'indirizzo di memoria reale dove NumPy tiene i dati dell'array
input_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Chiamata alla funzione effettiva
lib.process_image(input_ptr, output_ptr, srcWidth, srcHeight, channels, dstWidth, dstHeight)

print("Resize effettato con successo!!!")

# Stampa dell'immagine ottenuta
output_hwc = np.transpose(output, (1, 2, 0))  # da (C,H,W) a (H,W,C)
cv2.imshow('Pipeline Output', output_hwc)
cv2.waitKey(0)
cv2.destroyAllWindows()