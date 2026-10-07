#include <iostream>

/*
Riassunto del kernel resize_kernel

Obiettivo: ridimensionare un'immagine da srcWidth × srcHeight a dstWidth × dstHeight, con un thread per 
ogni pixel dell'immagine di output (nearest neighbor resize).

Passi del kernel:

Indicizzazione del thread — idx, idy calcolati da blockIdx/blockDim/threadIdx (2D): rappresentano la 
posizione del pixel destinazione di cui questo thread è responsabile.
Scale factor in floating point — scaleX = srcWidth / dstWidth (cast a float, mai a int, altrimenti si 
perde la parte decimale e il resize degenera in un semplice ritaglio).
Mappatura verso la sorgente — srcX = (int)(idx * scaleX), srcY = (int)(idy * scaleY): moltiplicazione 
prima in floating point, cast a intero dopo sul risultato finale — mai troncare lo scale factor a monte.
Controllo dei limiti — if (idx < dstWidth && idy < dstHeight): confronta le coordinate destinazione con 
le dimensioni destinazione (non quelle sorgente, perché i thread sono mappati sull'output).
Lettura da input — val = input[srcY * srcWidth + srcX]: conversione 2D→1D usando le coordinate sorgente 
(srcX, srcY) e la larghezza sorgente (srcWidth).
Scrittura su output (ultimo pezzo, da completare): stessa logica di conversione 2D→1D, ma con idx, idy 
e dstWidth.

Il principio chiave da ricordare: ogni volta che converti coordinate 2D→1D, la larghezza usata nella 
formula deve sempre corrispondere all'immagine a cui quell'indice si riferisce — mai mescolare larghezza 
sorgente con coordinate destinazione o viceversa.
*/

__global__ void resize_kernel(unsigned char *input, unsigned char *output, 
                              int srcWidth, int srcHeight, int channels,
                              int dstWidth, int dstHeight){
    
    // Calcolo indice del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Scala di resize dell'immagine
    float scaleX = (float)srcWidth / dstWidth;        
    float scaleY = (float)srcHeight / dstHeight;

    // Calcolo degli indici
    int srcX = (int)(idx * scaleX);
    int srcY = (int)(idy * scaleY);

    // Overflow check
    if(idx < dstWidth && idy < dstHeight){

        // Ciclo per i canali
        for(int c = 0; c < channels; c++){

            // Lettura del valore di input
            unsigned char val = input[(srcY * srcWidth + srcX) * channels + c];

            // Scrittura del pixel sul vettore di output
            output[(idy * dstWidth + idx) * channels + c] = val;

        }

    }

}



/*
Riassunto del kernel normalize_kernel

Obiettivo: convertire i valori dei pixel di un'immagine dal range intero 0-255 (formato standard di 
un'immagine 8-bit) al range decimale 0.0-1.0, formato richiesto dai modelli di machine learning per 
funzionare correttamente.

Perché serve: i modelli ML (quello che userai nella Fase 4 per il riconoscimento gesti) sono allenati 
aspettandosi input in un range piccolo e centrato — dare in pasto valori come 200 invece di 0.784 
produrrebbe risultati completamente sbagliati.

Differenza chiave rispetto al resize: qui non c'è nessuna mappatura tra coordinate diverse — ogni thread 
legge e scrive esattamente la stessa posizione nell'array (elaborazione "in place" concettualmente, anche 
se fisicamente su due array separati per i tipi diversi). Per questo:

Indicizzazione 1D, non 2D: l'immagine è trattata come un array piatto di pixel, senza bisogno di conoscere 
riga/colonna
Nessuna larghezza/altezza nella firma: serve solo totPixels, il numero totale di elementi

Scelte sui tipi, fatte insieme:

unsigned char *inPixels — i pixel in ingresso, fedeli al formato reale di un'immagine 8-bit (0-255 sta 
comodamente in un byte)
float *outPixels — l'output, che deve necessariamente essere decimale
Due array separati (non un vero "in place"), perché il tipo cambia tra ingresso e uscita — non potresti 
scrivere un float dentro un array di unsigned char

Attenzione da avere nel corpo (il pezzo che stiamo per scrivere): la divisione tra un intero (il pixel) e 
255 rischia la stessa trappola della divisione intera vista più volte — serve assicurarsi che il calcolo
avvenga in floating point, altrimenti il risultato tronca sempre a 0 o 1, perdendo tutta la scala 
intermedia.
*/

__global__ void normalize_kernel(unsigned char *inPixels, float *outPixels, int totPixels){

    // Calcolo indice
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Overflow check
    if(id < totPixels){
        
        // Lettura del pixel
        unsigned char px = inPixels[id];

        // Normalizzazione e conversione
        float pxNorm = (float)(px) / 255.0f;

        // Scrittura sul vettore di output
        outPixels[id] = pxNorm;
    }

}



/*
Riassunto veloce di questo kernel

Obiettivo: riorganizza un'immagine a colori dal formato HWC (canali intrecciati per pixel, come la 
fornisce OpenCV) al formato CHW (canali separati a blocchi, come lo richiede PyTorch) — stesso contenuto, 
ordine diverso in memoria.

Le due formule, il cuore del kernel:

HWC: (idy * width + idx) * channels + c — ogni pixel occupa channels celle consecutive
CHW: c * (width * height) + (idy * width + idx) — ogni canale occupa un intero blocco di width*height 
celle

Perché un ciclo for sui canali dentro il thread, invece di un terzo asse idz: con solo 3 canali, il 
parallelismo vero conta su righe/colonne (che possono essere migliaia), non su una dimensione fissa e 
minuscola come i canali.
*/

__global__ void HWCToCHW(float *input, float *output,
                          int width, int height, int channels){

    // Calcolo indice del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Overflow check
    if(idx < width && idy < height){

        // Ciclo che verrà usato per andare a considerare tutti i canali dell'immagine
        for(int c = 0; c < channels; c++){
        
            int hwcIndex = (idy * width + idx) * channels + c; // formula HWC (input)
            int chwIndex = c * (width * height) + (idy * width + idx);  // formula CHW (output)

            // Leggo il pixel dentro la matrice di input e lo scrivo su quella di output
            output[chwIndex] = input[hwcIndex];

        }

    }

}



int main(){

    // Dimensioni della matrice di input
    int dimSrcWidth = 8;
    int dimSrcHeight = 8;

    // Dimensione della matrice di output
    int dimDstWidth = 4;
    int dimDstHeight = 4;
    int channels = 3; // Canali dei colori della matrice
    int totPixels = dimDstWidth * dimDstHeight * channels;   // 8*8 = 64, i pixel USCITI dal resize

    // Calcolo delle size per allocazione in memoria
    size_t sizeInput = dimSrcWidth * dimSrcHeight * channels * sizeof(unsigned char);  // Input
    size_t sizeOutput = dimDstWidth * dimDstHeight * channels * sizeof(unsigned char); // Output
    size_t sizeOutNorm = dimDstWidth * dimDstHeight * channels * sizeof(float);

    // Definizione matrici
    unsigned char *h_matIn = (unsigned char *)malloc(sizeInput);    // Matrice di input
    unsigned char *h_matOut = (unsigned char *)malloc(sizeOutput);  // Matrice di output
    float *h_matNorm = (float *)malloc(sizeOutNorm);  // Matrice di output normalizzata
    float *h_matConverted = (float *)malloc(sizeOutNorm); // Matrice convertita HCW -> CHW

    // Allocazione della matrice di intput
    for(int row = 0; row < dimSrcHeight; row++){
        for(int col = 0; col < dimSrcWidth; col++){
            for(int c = 0; c < channels; c++){
                h_matIn[(row * dimSrcWidth + col) * channels + c] = (row * dimSrcWidth + col) * channels + c; 
            }
        }
    }



    // Definizione delle matrici su Device
    unsigned char *d_matIn, *d_matOut;
    float *d_matNorm, *d_matConverted;

    // Allocazione sul Device della memoria
    cudaMalloc(&d_matIn, sizeInput);
    cudaMalloc(&d_matOut, sizeOutput);
    cudaMalloc(&d_matNorm, sizeOutNorm);
    cudaMalloc(&d_matConverted, sizeOutNorm);

    // Copia della matrice di input da Host a Device
    cudaMemcpy(d_matIn, h_matIn, sizeInput, cudaMemcpyHostToDevice);



    // Definizione totale dei thread operativi su GPU per kernel 1
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((dimDstWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dimDstHeight + threadsPerBlock.y - 1) / threadsPerBlock.y); 

    // Chiamata al kernel di resize
    resize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matIn, d_matOut, 
                                                      dimSrcWidth, dimSrcHeight, channels,
                                                      dimDstWidth, dimDstHeight);

    // Sincronizzazione tra CPU e GPU
    cudaDeviceSynchronize();

    // Copia della matrice di output
    cudaMemcpy(h_matOut, d_matOut, sizeOutput, cudaMemcpyDeviceToHost);



    // Definizione dei thread per kernel 2
    int threadsPerBlockNorm = 256;
    int blocksPerGridNorm = (totPixels + threadsPerBlockNorm - 1) / threadsPerBlockNorm;

    // Chiamata al kernel di normalizzazione
    normalize_kernel<<<blocksPerGridNorm, threadsPerBlockNorm>>>(d_matOut, d_matNorm, totPixels);

    // Sincronizzazione tra CPU e GPU
    cudaDeviceSynchronize();

    // Copia della matrice da Device a Host
    cudaMemcpy(h_matNorm, d_matNorm, sizeOutNorm, cudaMemcpyDeviceToHost);



    // Chiamata al kernel 3 -> Riuso i thread inizializzati al kernel 1 alla fine la dimensione non cambia più
    HWCToCHW<<<blocksPerGrid, threadsPerBlock>>>(d_matNorm, d_matConverted, 
                                                 dimDstWidth, dimDstHeight, channels);

    // Sincronizzazione tra CPU e GPU
    cudaDeviceSynchronize();

    // Copia della matrice da Device a Host
    cudaMemcpy(h_matConverted, d_matConverted, sizeOutNorm, cudaMemcpyDeviceToHost);



    // Controllo di correttezza dell'intera pipeline (resize + normalizzazione + HWC->CHW)
    bool tuttoCorretto = true;
    for (int row = 0; row < dimDstHeight; row++) {
        for (int col = 0; col < dimDstWidth; col++) {
            for (int c = 0; c < channels; c++) {

                // Ricalcolo, sulla CPU, le stesse identiche trasformazioni fatte dai 3 kernel
                int srcX = (int)(col * ((float)dimSrcWidth / dimDstWidth));
                int srcY = (int)(row * ((float)dimSrcHeight / dimDstHeight));

                unsigned char pixelOriginale = h_matIn[(srcY * dimSrcWidth + srcX) * channels + c];
                float pixelNormalizzato = (float)(pixelOriginale) / 255.0f;

                int chwIndex = c * (dimDstWidth * dimDstHeight) + (row * dimDstWidth + col);
                float atteso = pixelNormalizzato;
                float ottenuto = h_matConverted[chwIndex];

                if (atteso != ottenuto) {
                    std::cout << "MISMATCH a (row=" << row << ", col=" << col << ", c=" << c
                            << "): atteso " << atteso << ", ottenuto " << ottenuto << std::endl;
                    tuttoCorretto = false;
                }
            }
        }
    }
    if (tuttoCorretto) {
        std::cout << "Pipeline completa corretta! Tutti i " 
                << (dimDstWidth * dimDstHeight * channels) 
                << " valori corrispondono." << std::endl;
    } else {
        std::cout << "Trovati errori nella pipeline." << std::endl;
    }



    // Pulizia della memoria Host
    free(h_matIn);
    free(h_matOut);
    free(h_matNorm);
    free(h_matConverted);

    // Pulizia della memoria Device
    cudaFree(d_matIn);
    cudaFree(d_matOut);
    cudaFree(d_matNorm);
    cudaFree(d_matConverted);

    return 0;
}