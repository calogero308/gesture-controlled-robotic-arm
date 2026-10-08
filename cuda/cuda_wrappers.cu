#include <cuda_runtime.h>

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




__global__ void resize_area_kernel(unsigned char *input, unsigned char *output, 
                              int srcWidth, int srcHeight, int channels,
                              int dstWidth, int dstHeight){

    // Calcolo indice del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Scala di resize dell'immagine
    float scaleX = (float)srcWidth / dstWidth;        
    float scaleY = (float)srcHeight / dstHeight;

    int xStart = (int)floorf(idx * scaleX);        
    int xEnd   = (int)ceilf((idx + 1) * scaleX);   
    
    int yStart = (int)floorf(idy * scaleY);
    int yEnd   = (int)ceilf((idy + 1) * scaleY);

        // Overflow check
    if(idx < dstWidth && idy < dstHeight){

        // Il blocco non deve uscire dall'immagine sorgente
        if(xEnd > srcWidth)  xEnd = srcWidth;
        if(yEnd > srcHeight) yEnd = srcHeight;

        // Ciclo per i canali
        for(int c = 0; c < channels; c++){

            int sum = 0;     // somma dei valori del blocco
            int count = 0;   // quanti pixel ci sono nel blocco

            for(int y = yStart; y < yEnd; y++){
                for(int x = xStart; x < xEnd; x++){

                    // Calcolo della somma dei Pixel
                    sum += input[(y * srcWidth + x) * channels + c];
                    count++;

                }
            }

            output[(idy * dstWidth + idx) * channels + c] = (unsigned char)(sum / (float)count + 0.5f);

        }
    }
    

}




__global__ void normalize_kernel(unsigned char *inPixels, float *outPixels,
                                 int totPixels, int channels){

    // Valori ImageNet, gli stessi di config.py (IMAGENET_MEAN / IMAGENET_STD)
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    // Calcolo indice
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Overflow check
    if(id < totPixels){

        // BUCO: a quale canale appartiene questo valore?
        int c = id % channels;

        // Normalizzazione: scala in [0,1], poi sottrae la media e divide per la deviazione standard
        float pxNorm = ((float)inPixels[id] / 255.0f - mean[c]) / stdv[c];

        // Scrittura sul vettore di output
        outPixels[id] = pxNorm;
    }
}




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




extern "C" void process_image(unsigned char* h_input, float* h_output,
                            int srcWidth, int srcHeight, int channels,
                            int dstWidth, int dstHeight) {

    // Calcolo delle size per allocazione in memoria
    size_t sizeInput = srcWidth * srcHeight * channels * sizeof(unsigned char);  // Input
    size_t sizeOutput = dstWidth * dstHeight * channels * sizeof(unsigned char); // Output
    size_t sizeOutNorm = dstWidth * dstHeight * channels * sizeof(float);
    int totPixels = dstWidth * dstHeight * channels;

    // Definizione delle matrici su Device
    unsigned char *d_matIn, *d_matOut;
    float *d_matNorm, *d_matConverted;

    // Allocazione sul Device della memoria
    cudaMalloc(&d_matIn, sizeInput);
    cudaMalloc(&d_matOut, sizeOutput);
    cudaMalloc(&d_matNorm, sizeOutNorm);
    cudaMalloc(&d_matConverted, sizeOutNorm);

    // Copia della matrice di input da Host a Device
    cudaMemcpy(d_matIn, h_input, sizeInput, cudaMemcpyHostToDevice);

    // Definizione totale dei thread operativi su GPU per kernel 1
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((dstWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dstHeight + threadsPerBlock.y - 1) / threadsPerBlock.y); 

    // Chiamata al kernel di resize
    resize_area_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matIn, d_matOut, 
                                                      srcWidth, srcHeight, channels,
                                                      dstWidth, dstHeight);

    // Sincronizzazione tra CPU e GPU
    cudaDeviceSynchronize();

     // Definizione dei thread per kernel 2
    int threadsPerBlockNorm = 256;
    int blocksPerGridNorm = (totPixels + threadsPerBlockNorm - 1) / threadsPerBlockNorm;

    // Chiamata al kernel di normalizzazione
    normalize_kernel<<<blocksPerGridNorm, threadsPerBlockNorm>>>(d_matOut, d_matNorm, totPixels, channels);

    // Sincronizzazione tra CPU e GPU
    cudaDeviceSynchronize();

    // Chiamata al kernel 3 -> Riuso i thread inizializzati al kernel 1 alla fine la dimensione non cambia più
    HWCToCHW<<<blocksPerGrid, threadsPerBlock>>>(d_matNorm, d_matConverted, 
                                                 dstWidth, dstHeight, channels);

    // Sincronizzazione tra CPU e GPU
    cudaDeviceSynchronize();

    // Copia della matrice da Device a Host
    cudaMemcpy(h_output, d_matConverted, sizeOutNorm, cudaMemcpyDeviceToHost);

    // Pulizia della memoria Device
    cudaFree(d_matIn);
    cudaFree(d_matOut);
    cudaFree(d_matNorm);
    cudaFree(d_matConverted);

}