# [Nome progetto da decidere] — Controllo gestuale di un robot con percezione CUDA-accelerata

## Cos'è

Un robot simulato (ROS2 + Gazebo) che attraversa un percorso a ostacoli guidato da
comandi gestuali della mano, con la pipeline di percezione (rilevamento gesti,
preprocessing immagini) accelerata su GPU tramite kernel CUDA scritti da zero.

Il progetto nasce per intrecciare in un unico sistema coerente tre aree distinte —
programmazione di sistema in C++/CUDA, robotica (ROS2), machine learning — invece
di trattarle come competenze separate.

## Come funziona

1. Una camera (reale o simulata) inquadra la mano dell'operatore
2. Un modello di riconoscimento gesti classifica il gesto corrente (avanti / stop /
   sinistra / destra — set esatto in definizione)
3. Il preprocessing dell'immagine (resize, normalizzazione, conversione layout) gira
   su kernel CUDA scritti interamente da zero, non su libreria esterna
4. Il gesto riconosciuto viene tradotto in un comando di navigazione ad alto livello
5. Il robot, in un nodo ROS2 dedicato, traduce il comando in movimento, evitando
   autonomamente gli ostacoli del percorso in modo reattivo (non appreso — vedi
   sezione "Sviluppi futuri")

## Stack tecnico

- **C++ / CUDA** — kernel di preprocessing immagini, logica di sistema
- **ROS2 (Jazzy)** — comunicazione tra i nodi (camera, inferenza, controllo robot)
- **Gazebo** — simulazione del robot e del percorso a ostacoli
- **PyTorch** — training/inferenza del modello di riconoscimento gesti
- **Python + C++** — la pipeline attraversa entrambi i linguaggi, con l'inferenza e
  il preprocessing pesante lato CUDA/C++, l'orchestrazione ROS2 dove più naturale

## Stato del progetto

In sviluppo — vedi il file di roadmap per l'elenco dettagliato delle fasi e
l'avanzamento corrente.

## Sviluppi futuri

- **Sostituire l'evitamento ostacoli reattivo con una policy di Reinforcement
  Learning**, addestrata in simulazione — pianificato per coincidere con il corso
  di Reinforcement Learning del prossimo semestre, in modo da applicare
  direttamente la teoria del corso a un sistema reale già funzionante invece che
  a un progetto giocattolo costruito ad hoc
- (altre idee da aggiungere strada facendo)

## Perché questo progetto

Costruito interamente da zero — nessun tutorial/repository copiato — proprio per
consolidare competenze che altrimenti resterebbero solo teoriche: gestione GPU a
basso livello, comunicazione tra nodi robotici, integrazione di un modello ML in un
sistema real-time.
