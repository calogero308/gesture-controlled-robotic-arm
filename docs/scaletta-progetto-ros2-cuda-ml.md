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
| ROS2 Jazzy | ❌ Da installare (sessione dedicata, 2-4 h stimate) | Fase 3 |
| Gazebo | ❌ Da installare | Fase 5 |

---

## FASE 1 — Preprocessing immagini in CUDA puro ✅ COMPLETATA

- [x] **Kernel di resize** (nearest neighbor, `unsigned char`, scale factor in `float`)
- [x] **Kernel di normalizzazione** (0-255 → 0.0-1.0, indicizzazione 1D)
- [x] **Kernel di conversione layout** HWC → CHW
- [x] Verifica di correttezza con confronto CPU nel `main` di test
- [x] Pulizia memoria (host + device) verificata

**Nota tecnica:** quando due kernel si incatenano con tipi diversi (`unsigned char` →
`float`), ogni buffer va allocato con la `size_t` calcolata sul *proprio* tipo.

---

## FASE 2 — Binding Python e OpenCV ✅ COMPLETATA (con un punto aperto)

- [x] Installare OpenCV Python
- [x] Wrapper `extern "C"` (`cuda_wrappers.cu`, compilato in `.so`) e binding con `ctypes`
- [x] `process_image`: resize → normalize → HWC→CHW interamente sulla GPU, verificato su
      una foto reale
- [ ] Confronto numerico con `cv2.resize` / normalizzazione OpenCV (finora solo
      verifica visiva) — da fare nell'integrazione, insieme al punto resize della Fase 4

---

## FASE 3 — ROS2 puro (Docker + ROS2 Jazzy) ⏸ RIMANDATA

Sessione dedicata. Nessun collegamento con CUDA/ML: solo il meccanismo di comunicazione.

- [ ] Dockerfile unico ROS2 + CUDA, GPU visibile nel container
- [ ] Nodo "hello world": publisher/subscriber, capire bene il meccanismo
- [ ] Nodo "camera finta": publisher di immagini (`sensor_msgs/Image`)
- [ ] Nodo "subscriber immagine": riceve e verifica il canale

---

## FASE 4 — Riconoscimento gesti della mano ✅ MODELLO COMPLETATO, integrazione da fare

**Gesti (5 classi, dataset HaGRID):** fist, like, palm, peace, rock — 34.603 immagini.
Mappatura gesto → comando del braccio: da definire.

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
- [ ] **Integrazione con la pipeline CUDA** — tre disallineamenti da sistemare:
      1. normalizzazione: il training usa `(x - mean) / std` di ImageNet, il
         `normalize_kernel` fa solo `x / 255`
      2. ordine dei canali: OpenCV legge BGR, il training è RGB
      3. resize: nearest-neighbor nel kernel contro bilineare in torchvision —
         verifica numerica
- [ ] Ritaglio della mano (o regione fissa): il dataset contiene ritagli stretti, con
      scene intere l'accuratezza cala
- [ ] Benchmark: tempo di inferenza con/senza preprocessing CUDA custom
- [ ] (Miglioramenti futuri del modello) augmentation più aggressiva (zoom/ritaglio
      casuale, prospettiva), più epoche, eventuali foto proprie nel training; voto a
      maggioranza su più frame per ridurre gli errori sporadici

**Note tecniche da ricordare:**
- Congelare tutto il backbone non bastava: servivano gli ultimi blocchi sbloccati.
- Non mettere cartelle extra (es. `test/`) dentro la radice letta da `ImageFolder`: verrebbero
  contate come classi aggiuntive e romperebbero il training (il layer finale ha 5 uscite).
- Una confidenza alta non garantisce una predizione corretta: la soglia da sola non basta.

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

- [ ] README aggiornato: braccio al posto del robot mobile, risultati ML (74% → 97.3%),
      limiti noti (peace/rock, dominio ritagli), storia delle scelte
- [ ] Commit Git progressivi per blocco (ML, integrazione, braccio) con `.gitignore` per
      `models/*.pth`, `gesti_estratti/`, `data/`
- [ ] Gif/video breve del sistema in azione
- [ ] Collegamento del repository su LinkedIn/CV, con una riga che spiega
      l'intreccio C++/CUDA + ROS2 + ML + cinematica

---

## Note aperte

- Mappatura esatta gesto → direzione/azione del braccio (5 gesti disponibili)
- Parametri DH: braccio inventato da me o ispirato a un modello reale
- Webcam reale (accesso da WSL2 non immediato) o camera simulata in Gazebo
- Nome definitivo del progetto e del repository