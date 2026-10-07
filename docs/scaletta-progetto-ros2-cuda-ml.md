# Progetto Specializzazione — ROS2 + CUDA + ML per il mercato svizzero
## Percorso "da zero", costruito pezzo per pezzo, per portfolio GitHub/LinkedIn

**Obiettivo:** un robot simulato che attraversa un percorso a ostacoli guidato da
comandi gestuali della mano, con percezione CUDA-accelerata — un progetto unico e
coerente (non tre progetti separati) che dimostri l'intreccio di C++/CUDA, ROS2 e ML,
esattamente le competenze richieste per ruoli robotica/automazione in Svizzera.

**Evitamento ostacoli:** reattivo (non appreso) nella v1. La sostituzione con una
policy di Reinforcement Learning è pianificata come v2, per coincidere con il corso
di RL del prossimo semestre — vedi README del progetto per i dettagli.

**Hardware:** RTX 5050 (8GB VRAM), 32GB RAM, i7 — più che sufficiente per l'intero progetto.

**Regola del percorso:** ogni pezzo scritto da zero (niente copia di tutorial/repo
esistenti), commit progressivi visibili su GitHub, non un unico commit finale.

---

## Checklist ambiente — cosa ho, cosa manca

| Componente | Stato al 15/09/2026 | Quando serve |
|---|---|---|
| CUDA Toolkit (nvcc 12.6) | ✅ Già installato | Da subito |
| Python 3.12 + pip | ✅ Già installato | Da subito |
| Ubuntu 24.04 (WSL) | ✅ Confermato | — |
| ROS2 Jazzy | ❌ Da installare | Fase 3 |
| PyTorch (con supporto CUDA) | ❌ Da installare | Fase 4 |
| OpenCV | ❌ Da installare | Fase 2-4 |
| Gazebo | ❌ Da installare | Fase 5 |

Stima download totale rimanente: ~5-5.5 GB (ROS2 desktop ~2.5-3GB, PyTorch ~2-2.5GB,
OpenCV+resto ~300-500MB, Gazebo quando serve ~1-1.5GB).

---

## FASE 1 — Preprocessing immagini in CUDA puro (in corso, zero installazioni nuove)

Nessuna dipendenza esterna oltre a quello già presente. Obiettivo: costruire i kernel
che serviranno da ponte tra "immagine grezza" e "input pronto per il modello ML",
riusabili indipendentemente da quale libreria ML si userà poi.

- [x] **Kernel di resize** — ridimensiona un'immagine da una risoluzione a un'altra
      (nearest neighbor). Tipi: `unsigned char*` in ingresso/uscita (fedele al
      formato reale di un'immagine 8-bit, non `float` come nel primo tentativo),
      scale factor e coordinate intermedie sempre in `float` per evitare che il
      resize degeneri in un crop. Verificato con controllo automatico su tutti i
      pixel (confronto con formula ricalcolata sulla CPU) — nessun mismatch.
- [x] **Kernel di normalizzazione** — porta i valori pixel da 0-255 (`unsigned char`)
      a 0.0-1.0 (`float`). Indicizzazione 1D (nessuna larghezza/altezza necessaria,
      a differenza del resize). Collegato in catena diretta dopo il resize (il
      buffer di output del resize diventa l'input della normalizzazione, stesso
      `main`, senza passaggi intermedi). Eseguito senza crash dopo la correzione
      della dimensione di allocazione device (era calcolata sul tipo sbagliato).
- [ ] **Kernel di conversione layout** — da HWC (height-width-channel) a CHW
      (concettualmente simile alla trasposizione di matrice già fatta per l'esame)
- [x] Verifica di correttezza — fatta per il resize (confronto CPU su tutti i pixel);
      normalizzazione verificata solo per esecuzione senza errori, non ancora con
      un controllo numerico esplicito (da aggiungere se si riprende in mano il file)
- [x] Pulizia memoria (host + device) per entrambi i kernel — verificata completa

**Prossimo compito concreto:** kernel di conversione layout HWC→CHW.

**Nota tecnica da ricordare:** quando due kernel si incatenano con tipi diversi
lungo la pipeline (qui: resize in `unsigned char` seguito da normalizzazione che
produce `float`), ogni buffer va allocato con la `size_t` calcolata sul *proprio*
tipo — un errore fatto una volta in questo progetto è stato riusare la size di un
buffer `unsigned char` per allocare quello `float` corrispondente, causando
un'allocazione troppo piccola (e potenziale accesso fuori dai limiti).

---

## FASE 2 — Introduzione a OpenCV (installazione minima)

Solo quando la Fase 1 è solida. Serve per caricare immagini vere (non più solo dati
sintetici) e confrontare il proprio kernel di preprocessing con l'equivalente OpenCV,
per validare che il proprio codice CUDA produca risultati corretti su dati reali.

- [ ] Installare OpenCV Python (`pip install opencv-python`)
- [ ] Caricare un'immagine reale, passarla al proprio kernel CUDA (via Python
      binding o semplice scambio file, da decidere insieme quando ci si arriva)
- [ ] Confrontare l'output del proprio kernel con `cv2.resize`/normalizzazione OpenCV

---

## FASE 3 — ROS2 puro (installazione ROS2 Jazzy)

Nessun collegamento ancora con CUDA/ML — solo il meccanismo di comunicazione.

- [ ] Installare ROS2 Jazzy Desktop (comandi già pronti quando si arriva qui)
- [ ] Nodo "hello world": un publisher che manda numeri incrementali, un subscriber
      che li riceve e stampa — capire bene il meccanismo pub/sub prima di aggiungere
      complessità
- [ ] Nodo "camera finta": un publisher che pubblica un'immagine statica (o un frame
      da webcam) a intervalli regolari, sul formato messaggio giusto per immagini
      (`sensor_msgs/Image`)
- [ ] Nodo "subscriber immagine": riceve l'immagine pubblicata e la salva/stampa le
      dimensioni, per verificare che il canale funzioni

---

## FASE 4 — Riconoscimento gesti della mano, in inferenza (installazione PyTorch)

Set di gesti da definire (indicativamente 4-5: avanti, stop, sinistra, destra, +
eventuale "modalità" extra) — meno gesti = training più veloce e affidabile per
iniziare, espandibile in seguito.

- [ ] Installare PyTorch con supporto CUDA (`torch.cuda.is_available()` deve dare `True`)
- [ ] Raccogliere/generare un piccolo dataset di immagini per gesto (webcam propria)
- [ ] Allenare un classificatore semplice sui gesti definiti (il rilevamento dei
      keypoint della mano può appoggiarsi a uno strumento esistente come MediaPipe,
      ma la classificazione del gesto sopra quei keypoint è lavoro proprio)
- [ ] Sostituire il preprocessing "standard" con i propri kernel CUDA della Fase 1
      (qui i due pezzi iniziano davvero a intrecciarsi)
- [ ] Misurare e confrontare il tempo di inferenza con/senza il preprocessing custom

---

## FASE 5 — Integrazione completa in un package ROS2

Tutto insieme: il pezzo che rende il progetto "uno", non tre cose affiancate.

- [ ] Nodo camera (Fase 3) → nodo inferenza gesti (Fase 4, preprocessing CUDA +
      modello) → nodo controllo robot (traduce il gesto in comando di movimento)
- [ ] Package ROS2 strutturato correttamente (CMakeLists.txt o setup.py, package.xml)
- [ ] Mondo Gazebo con percorso a ostacoli (alcuni box/oggetti posizionati)
- [ ] Evitamento ostacoli reattivo (basato su sensori del robot, non appreso — la
      versione appresa è pianificata come v2, vedi README)

---

## FASE 6 — Rifinitura portfolio

- [ ] README che racconta la storia: problema → scelte tecniche → perché quella
      scelta (non solo "cosa fa" il codice)
- [ ] Commit history pulita e progressiva (verificabile: non un solo commit gigante)
- [ ] Se possibile, una gif/video breve del sistema in azione (Gazebo o webcam reale)
- [ ] Collegamento del repository su LinkedIn/CV, con una riga che spiega
      esplicitamente l'intreccio C++/CUDA + ROS2 + ML

---

## Note aperte da chiarire strada facendo

- Numero e set esatto di gesti da riconoscere (indicativamente 4-5)
- Se procedere con webcam reale o solo immagini statiche/dataset per i test dei gesti
- Nome definitivo del progetto e del repository
