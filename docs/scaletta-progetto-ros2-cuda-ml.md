# Progetto Specializzazione — ROS2 + CUDA + ML
## Percorso "da zero", costruito pezzo per pezzo, per portfolio GitHub/LinkedIn

**Obiettivo:** un **braccio robotico simulato (6 DOF con polso sferico)** controllato da
gesti della mano, con percezione CUDA-accelerata e cinematica inversa scritta da me —
un progetto unico e coerente (non tre progetti separati) che dimostri l'intreccio di
C++/CUDA, ROS2 e ML, esattamente le competenze richieste per ruoli di robotica e
automazione.

**Cambio di rotta (30/09/2026):** il robot mobile con percorso a ostacoli è sostituito dal
braccio, per collegare il progetto al corso di robotica industriale (cinematica del
polso sferico). Tutta la pipeline CUDA + ML resta identica: cambia solo cosa riceve il
comando finale. Movimenti semplici (destra/sinistra/su/giù), ma **calcolati al volo**
con cinematica inversa (disaccoppiamento posizione/orientamento), non posizioni
pre-salvate.

**Reinforcement Learning:** rimandato a una v2, per coincidere con il corso di RL del
prossimo semestre. Da ridefinire per il caso braccio — vedi README.

**Hardware:** RTX 5050 (8GB VRAM), 32GB RAM, i7 — più che sufficiente per l'intero progetto.

**Regola del percorso:** ogni pezzo scritto da zero (niente copia di tutorial/repo
esistenti; il dataset pubblico è ammesso, il codice no), commit progressivi visibili su
GitHub, non un unico commit finale.

---

## Checklist ambiente

| Componente | Stato al 07/10/2026 | Quando serve |
|---|---|---|
| CUDA Toolkit (nvcc 12.6) | ✅ Installato | — |
| Python 3.12 + pip | ✅ Installato | — |
| Ubuntu 24.04 (WSL2) | ✅ Confermato | — |
| Docker Desktop + GPU passthrough | ✅ Verificato | Fase 3 |
| PyTorch (build cu130, per RTX 5050 / sm_120) | ✅ Installato | — |
| OpenCV | ✅ Installato | — |
| `huggingface_hub` (solo per lo script di download del dataset) | ✅ Usato | — |
| ROS2 Jazzy | ❌ Da installare (sessione dedicata, 2-4 h stimate) | Fase 3 |
| Gazebo | ❌ Da installare | Fase 5 |

---

## FASE 1 — Preprocessing immagini in CUDA puro ✅ COMPLETATA

- [x] **Kernel di resize** (nearest neighbor, `unsigned char`, scale factor in `float`) —
      poi affiancato in Fase 4 da `resize_area_kernel` (media dell'area), che è quello
      usato dalla pipeline
- [x] **Kernel di normalizzazione** (0-255 → 0.0-1.0, indicizzazione 1D) — esteso in Fase 4
      con mean/std ImageNet per canale
- [x] **Kernel di conversione layout** HWC → CHW
- [x] Verifica di correttezza con confronto CPU nel `main` di test
- [x] Pulizia memoria (host + device) verificata

**Nota tecnica:** quando due kernel si incatenano con tipi diversi (`unsigned char` →
`float`), ogni buffer va allocato con la `size_t` calcolata sul *proprio* tipo.

---

## FASE 2 — Binding Python e OpenCV ✅ COMPLETATA

- [x] Installare OpenCV Python
- [x] Wrapper `extern "C"` (`cuda_wrappers.cu`, compilato in `.so`) e binding con `ctypes`
- [x] `process_image`: resize → normalize → HWC→CHW interamente sulla GPU, verificato su
      una foto reale
- [x] Confronto numerico: fatto contro la pipeline torchvision del training (il
      riferimento che conta per il modello), non contro `cv2.resize` — vedi Fase 4

---

## FASE 3 — ROS2 puro (Docker + ROS2 Jazzy) ⏸ RIMANDATA

Sessione dedicata. Nessun collegamento con CUDA/ML: solo il meccanismo di comunicazione.

- [ ] Dockerfile unico ROS2 + CUDA, GPU visibile nel container
- [ ] Nodo "hello world": publisher/subscriber, capire bene il meccanismo
- [ ] Nodo "camera finta": publisher di immagini (`sensor_msgs/Image`)
- [ ] Nodo "subscriber immagine": riceve e verifica il canale

---

## FASE 4 — Riconoscimento gesti e integrazione con CUDA ✅ COMPLETATA (restano ritaglio mano e miglioramenti futuri)

**Gesti (5 classi, dataset HaGRID):** fist, like, palm, peace, rock — 34.603 immagini.
Mappatura gesto → comando del braccio: da definire.

### Modello

- [x] PyTorch con supporto CUDA (`torch.cuda.is_available()` = True)
- [x] Dataset pubblico HaGRID, filtrato su 5 classi
- [x] `dataset.py`: ImageFolder, split 80/20 (seed 42), augmentation (rotazione ±15°,
      luminosità/contrasto)
- [x] `train.py`: MobileNetV2 pretrained, validazione ad ogni epoca, checkpoint migliore
      - backbone congelato: **74%** (underfitting)
      - fine-tuning degli ultimi blocchi (`features[15:]`, lr 1e-4 sul backbone e 1e-3
        sul layer finale): **97.3%** val acc
- [x] `evaluate.py`: matrice di confusione — errore principale peace↔rock (90 errori su 184)
- [x] `inference.py`: ricostruisce il modello, carica `models/best_model_finetuned.pth`,
      restituisce gesto e confidenza
- [x] Test su foto personali (`data/test/`): **14/15 corrette**. Errore: un `rock` letto
      come `like` al 96% di confidenza (foto dal basso, mignolo non visibile, pollice
      esteso, volto dominante nell'inquadratura). Peace e rock restano i gesti più deboli.

### Integrazione con la pipeline CUDA (07/10/2026)

- [x] `cuda_preprocess.py`: binding Python con controlli su dtype, shape (H, W, 3) e
      contiguità dell'array prima di passare i puntatori alla GPU
- [x] I tre disallineamenti sistemati:
      1. normalizzazione: `normalize_kernel` ora applica `(x/255 - mean[c]) / std[c]` di
         ImageNet per canale (`c = id % channels`); nessun calo di accuratezza rispetto
         alla pipeline torchvision
      2. ordine dei canali: la pipeline CUDA si aspetta RGB (come il training). I frame
         OpenCV (BGR) vanno convertiti all'acquisizione: da applicare nel nodo camera,
         non ancora testato con una webcam reale
      3. resize: il nearest-neighbor (campionamento all'angolo del pixel, mentre PIL
         campiona al centro) dava un errore medio di 0.038 contro il bilineare con
         antialias di torchvision su una foto 1200×1600; sostituito con
         `resize_area_kernel` (media dell'area sorgente per ogni pixel di uscita):
         errore medio 0.004
- [x] Confronto numerico con torchvision (`compare_preprocessing.py`, `compare_batch.py`):
      su 500 immagini di validation **stessa classe in 495/500**, accuratezza **97.4%**
      (CUDA) contro 97.6% (torchvision), cioè 1 immagine di differenza
- [x] Benchmark (`benchmark.py`, RTX 5050 Laptop, 3 run, immagini sintetiche, mediana di
      200 ripetizioni dopo riscaldamento). Forward del modello: ~2.7 ms.

      | Input | CPU (ms) | CUDA (ms) | Speedup |
      |---|---|---|---|
      | 512×512 | 1.1–1.3 | 0.9–1.0 | 1.0×–1.3× |
      | 640×480 | 1.4–1.6 | 0.8–0.9 | 1.5×–1.9× |
      | 1920×1080 | 5.1–5.5 | 1.3–1.4 | 3.8×–4.2× |

      Il tempo CUDA cresce poco con la risoluzione: probabile costo fisso per chiamata
      (cudaMalloc/cudaFree, copie, sincronizzazioni) — inferenza, non ancora profilato.
- [x] README riscritto per il braccio, con risultati e limiti (vedi Fase 6)

### Ancora aperto

- [ ] **Ritaglio della mano (ROI)** davanti al classificatore: il dataset contiene
      ritagli stretti, con scene intere l'accuratezza cala (confermato: una foto intera
      `rock2` è letta `like` al 96% da entrambe le pipeline, quindi non è colpa del
      preprocessing)
- [ ] (Miglioramenti futuri del modello) augmentation più aggressiva (zoom/ritaglio
      casuale, prospettiva), più epoche, eventuali foto proprie nel training; voto a
      maggioranza su più frame per ridurre gli errori sporadici
- [ ] (Ottimizzazioni CUDA, v2) buffer GPU persistenti invece di allocarli a ogni
      chiamata, memoria host pinned, meno sincronizzazioni; eventuale kernel bilineare
      con antialias per chiudere il gap residuo con torchvision; profilazione con `nsys`
- [ ] Ispezionare i 5 casi su 500 in cui le due pipeline divergono (probabilmente casi
      al limite peace/rock)
- [ ] Testare il nuovo `download_dataset.py` (download ed estrazione non ancora provati)
- [ ] Pulizia: `python/test_resize.py` ha percorsi relativi vecchi; `cuda/resize_kernel.cu`
      duplica i kernel di `cuda_wrappers.cu` (serve come test standalone)

**Note tecniche da ricordare:**
- Congelare tutto il backbone non bastava: servivano gli ultimi blocchi sbloccati.
- Non mettere cartelle extra (es. `test/`) dentro la radice letta da `ImageFolder`: verrebbero
  contate come classi aggiuntive e romperebbero il training (il layer finale ha 5 uscite).
- Una confidenza alta non garantisce una predizione corretta: la soglia da sola non basta.
- Rilanciare `train.py` sovrascrive il checkpoint in `models/` (il salvataggio parte da
  `best_val_acc = 0`): copiare prima quello attuale.
- Con fattori di riduzione alti (es. 1200×1600 → 224×224) il nearest-neighbor fa aliasing:
  ogni pixel di uscita guarda 1 pixel su decine. Il confronto utile è sempre con la
  pipeline usata in training, non con una formula "ideale".
- Il tempo di una misura dipende dalle dimensioni: la pipeline CUDA conviene dal 640×480 in
  su; a 512×512 è alla pari con la CPU.

---

## FASE 5 — Braccio robotico e integrazione completa in ROS2

- [ ] Definire i parametri DH del braccio (6 DOF, polso sferico) — scelti da me
- [ ] Cinematica diretta (verifica) e **cinematica inversa analitica** con disaccoppiamento
      posizione/orientamento, scritta da zero
- [ ] Mappatura gesto → delta cartesiano (destra/sinistra/su/giù); IK calcolata al volo
- [ ] Modello URDF/Xacro del braccio, simulazione in Gazebo con `ros2_control`
- [ ] Nodo camera → nodo inferenza gesti (preprocessing CUDA + modello) → nodo controllo
      braccio (gesto → delta → IK → comandi ai giunti)
- [ ] Package ROS2 strutturato (CMakeLists.txt o setup.py, package.xml)

---

## FASE 6 — Rifinitura portfolio

- [x] README aggiornato: braccio al posto del robot mobile, risultati ML (74% → 97.3%),
      limiti noti (peace/rock, dominio ritagli), risultati dell'integrazione CUDA e
      benchmark
- [x] `.gitignore` per `models/*.pth`, `data/`, `*.so`
- [x] Commit Git progressivi per blocco (ML, integrazione CUDA, benchmark, README) —
      da proseguire per le fasi successive (braccio)
- [x] **Push su GitHub** (08/10/2026). Controllato: nessuna immagine, modello, dataset o
      `.so` né nei file attuali né nella cronologia (`data/` è ignorata)
- [ ] Prima di rendere il repo **pubblico**: confermare che sia privato nel frattempo e
      decidere l'email dei commit (noreply); cambiarla nei commit già fatti richiede di
      riscrivere la cronologia e un push forzato
- [ ] Gif/video breve del sistema in azione
- [ ] Collegamento del repository su LinkedIn/CV, con una riga che spiega
      l'intreccio C++/CUDA + ROS2 + ML + cinematica

---

## Note aperte

- Mappatura esatta gesto → direzione/azione del braccio (5 gesti disponibili)
- Parametri DH: braccio inventato da me o ispirato a un modello reale
- Webcam reale (accesso da WSL2 non immediato) o camera simulata in Gazebo
- Come ottenere il ritaglio della mano: regione fissa, detector leggero o segmentazione
- Nome definitivo del progetto e del repository
