# Hardware del braccio RRR: lista componenti (versione economica)

Lista di ciò che serve per assemblare il braccio fisico a 3 gradi di libertà (RRR), supponendo che **tutte le parti strutturali siano già stampate in 3D**. È la versione economica; quella con servo migliori è descritta in fondo.

I prezzi sono **ordini di grandezza in euro**, da verificare nel negozio scelto (non sono preventivi). Le quantità di viteria e cuscinetti vanno confermate sul modello CAD definitivo.

## Assunzioni di progetto

- Geometria da `python/arm/arm_config.py`: spalla a H = 0,09 m dal tavolo, braccio L1 = 0,16 m, avambraccio L2 = 0,14 m.
- Giunti: base θ1 ∈ [−90°, 90°], spalla θ2 ∈ [0°, 90°], gomito θ3 ∈ [−150°, −5°].
- Nessuna pinza (3 DOF). Se la si aggiunge serve un servo piccolo in più (MG90S, circa 3 €).
- Il PC (con la percezione e la cinematica) comanda l'ESP32 via USB seriale, più avanti via Wi-Fi o micro-ROS. Gli angoli sono calcolati dal PC.

## Stima della coppia (perché bastano servo da 10 kg·cm)

Caso peggiore: braccio disteso in orizzontale, senza carico, link da circa 50–60 g, servo del gomito da circa 60 g.

- Spalla: circa 0,25 N·m, cioè circa 2–3 kg·cm.
- Un MG996R dichiara circa 10 kg·cm di coppia di stallo: la spalla lavora a circa un quarto del limite.
- Con un carico o una pinza il valore sale in fretta e il conto va rifatto con i pesi reali (pesare le parti stampate e i servo).

È una stima, non una misura: serve a scegliere l'ordine di grandezza dei servo, non a certificarli.

## Elettronica

| Componente | Q.tà | Prezzo | Perché serve |
|---|---|---|---|
| Servo MG996R | 3 | circa 21 € | Un servo per giunto (base, spalla, gomito). Coppia sufficiente secondo la stima sopra, costo basso, comando PWM semplice. Limiti: nessun feedback di posizione, gioco negli ingranaggi, possibili vibrazioni. |
| ESP32 DevKit (30 pin) | 1 (meglio 2) | 6–12 € | Microcontrollore che genera i segnali PWM dei servo e riceve gli angoli dal PC. Ha molti pin PWM, Wi-Fi e Bluetooth integrati, e permette in futuro un nodo micro-ROS. Il secondo è di scorta. |
| Alimentatore 6 V, 5 A con spinotto DC | 1 | circa 15 € | I servo non vanno mai alimentati da USB o dal pin 5 V dell'ESP32: in movimento assorbono correnti molto più alte. 6 V dà più coppia dei 5 V; 5 A coprono i picchi di tre servo. |
| Adattatore da spinotto DC a morsetti a vite | 1 | circa 2 € | Collega l'alimentatore ai cavi dei servo senza tagliarlo. |
| Condensatore elettrolitico 1000 µF, 16 V | 2 | circa 1 € | Messo vicino ai servo assorbe i picchi di corrente e riduce i reset dell'ESP32 e le vibrazioni. |
| Fusibile 5 A con portafusibile | 1 | circa 2 € | Protegge alimentatore e cavi se un servo si blocca o c'è un corto. |
| Breadboard o basetta forata | 1 | circa 3 € | Distribuzione di alimentazione e massa; la basetta forata (saldata) è più affidabile della breadboard quando il braccio si muove. |
| Cavi Dupont (M-F e M-M) | 1 kit | circa 4 € | Collegamenti dei segnali tra ESP32 e servo. |
| Cavo USB dati per ESP32 | 1 | circa 3 € | Programmazione e comunicazione col PC. Deve essere un cavo dati, non solo di ricarica. |

## Meccanica

| Componente | Q.tà | Prezzo | Perché serve |
|---|---|---|---|
| Kit viti M3 assortite (8–25 mm) + dadi | 1 | circa 10 € | Fissano servo, link e coperchi. Le misure esatte si decidono sul CAD. |
| Inserti filettati a caldo M3 | 1 kit | circa 8 € | Danno filetti robusti nelle parti stampate, che si rovinano se si avvita direttamente nella plastica. Servono solo se il progetto li prevede. |
| Cuscinetti 608 (o quelli previsti dal CAD) | 3–6 | circa 6 € | Sostengono i carichi sui giunti, così l'albero del servo non fa da supporto strutturale. |
| Frenafiletti (Loctite blu) | 1 | circa 5 € | Evita che le viti si allentino con le vibrazioni dei servo. |
| Fascette e nastro biadesivo | 1 | circa 3 € | Fissaggio dei cavi lungo il braccio, per non farli impigliare nei giunti. |
| Morsetti da tavolo o base pesante | 2 | circa 8 € | La base deve essere ferma: se il braccio si sposta o si ribalta, la cinematica non conta. |

## Strumenti (se non già in casa)

| Strumento | Prezzo | Perché serve |
|---|---|---|
| Kit saldatore 60 W regolabile + stagno + treccia dissaldante | circa 25 € | Pin dell'ESP32, condensatori, connettori e cavi saldati reggono meglio dei fili infilati in una breadboard. Salda in un ambiente arieggiato. |
| Multimetro | circa 15 € | Controllare tensioni, continuità e massa in comune prima di accendere. |
| Cacciaviti di precisione e brugole | circa 10 € | Viti piccole e viti dei servo. |
| Pinze e tronchesino | circa 10 € | Spelare e tagliare cavi. |

## Totale indicativo

- Elettronica + meccanica: circa 80–90 €.
- Con gli strumenti: circa 150 €.
- Margine consigliato per spedizione, scarti e pezzi di scorta: +15 %.

## Collegamenti e precauzioni

1. I servo si alimentano **solo** dall'alimentatore a 6 V.
2. La **massa dell'alimentatore va collegata alla massa dell'ESP32** (massa in comune), altrimenti il segnale PWM non è riferito correttamente.
3. Il condensatore da 1000 µF va in parallelo all'alimentazione dei servo, vicino ai morsetti.
4. Il segnale PWM esce dai pin dell'ESP32 (logica a 3,3 V, di solito sufficiente per i servo); evitare i pin con funzione di avvio.
5. **Prima accensione senza il braccio montato**: servo liberi sul tavolo, per controllare direzione e angoli.
6. Nel firmware si riusano i limiti di `arm_config.py`: lo stesso controllo che in simulazione rifiuta i punti irraggiungibili protegge anche servo e meccanica.
7. Lo zero del servo e lo zero dell'angolo del modello non coincidono, quindi serve una calibrazione per giunto (θ1 usa bene i 180° del servo; per spalla e gomito serve un offset).

## Possibili miglioramenti (livelli superiori)

- **Servo digitali tipo DS3218** al posto degli MG996R: coppia e precisione maggiori, costo più alto (circa 18 € l'uno).
- **Encoder magnetici AS5600** per leggere l'angolo reale dei giunti anche con servo senza feedback (circa 5 € l'uno).
- **Servo smart seriali** (Feetech STS3215, Dynamixel XL430): feedback di posizione, corrente e temperatura, integrazione più pulita con ros2_control. Con i Dynamixel serve anche un adattatore USB per il bus.
- **Pinza** con un servo piccolo.

## Ordine di acquisto consigliato

1. Subito: ESP32, 1–2 servo economici di prova, alimentatore, cavi, condensatori, saldatore e multimetro. Si prova il controllo di un singolo giunto con la cinematica già scritta.
2. A modello finito: viteria, inserti, cuscinetti e base, quando le misure sono certe.
3. Solo dopo i test: valutare servo migliori, se serve più precisione.
