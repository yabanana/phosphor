# STBN 8×8×16: verifica spettrale CPU

## Protocollo congelato prima dei dati

**Stato iniziale: PREREGISTERED, nessuna generazione o FFT eseguita.**
Il commit di preregistrazione precede qualsiasi esecuzione del generatore.
Protocollo leggibile dalla macchina: `tools/testdata/stbn/protocol-v1.json`.
Sorgente congelato: `8e533bfec3254dca2684de539470ca8ab4e0ea55`, generatore v1.
Nessuna modifica a `stochastic_sampling.h/.cpp`; nessuna build del renderer,
GPU, profiler o installazione di runtime. Le misure saranno riprodotte in un
build locale del solo strumento CPU.

**Ipotesi:** il generatore riduce l'energia alle basse frequenze sia nelle
sezioni spaziali XY sia nelle storie temporali Z, mantenendo distribuzione
uniforme dei ranghi e senza forti correlazioni fra dimensioni. Non si deduce
questa proprietà dal nome STBN, né da una semplice permutazione corretta.

Configurazione esatta: 8×8×16, 64 dimensioni, sigma XY/Z 1,9, massimo 4.096
scambi di rilassamento. Otto seed prefissati: **0,1,2,3,4,5,6,7**. Si analizza
la tabella originale `(rank+0.5)/1024`, senza rotazione runtime. La rotazione
per blocco in sviluppo in un'altra chat resta fuori da questa prova.

Controlli indipendenti dal generatore C++: per ogni dimensione una permutazione
uniforme dei 1.024 ranghi (stessa marginale esatta) e campioni uniformi IID.
Si usa NumPy PCG64 con SeedSequence specificata nel JSON, senza usare
`stochasticHash` del motore. Le versioni effettive saranno registrate.

Si analizzano i pattern binari ai livelli **0,1 / 0,25 / 0,5 / 0,75 / 0,9**:

- Spazio: FFT2 di ogni sezione XY, banda `0 < kx²+ky² ≤ 2` (8 modi).
- Tempo: FFT di ogni pixel fisso attraverso i 16 frame, banda `|kt|=1`
  (2 modi). Non includiamo `|kt|=2`: a densità 0,1 vi sono mediamente 1,6
  punti per storia e quella frequenza può già appartenere all'anello atteso,
  anziché alla regione di basse frequenze da sopprimere.
- Nessuna finestra: la tabella è toroidale. DC rimossa per singola sezione/storia.
  Per seed si divide la potenza totale nella banda per la potenza totale
  non-DC, aggregando le 64 dimensioni e tutte le sezioni/storie. Le sezioni
  costanti contribuiscono zero a entrambi i termini; il loro numero è riportato.

**Soglie fissate ora:** per ciascun livello, asse e controllo, media su otto
seed del rapporto STBN/controllo **≤0,75**, e STBN migliore del controllo in
**almeno 7 seed su 8**. Non si aggregano livelli per nascondere una densità
sfavorevole. Inoltre: ranghi permutazione esatta e istogramma a 32 bin esatto
in ogni dimensione; rilassamento convergente per tutti i seed; correlazioni
Pearson assolute fra tutte le coppie di dimensioni, sui ranghi grezzi,
**p95≤0,10 e massimo≤0,25 per seed**. Queste ultime sono soglie operative
contro dimensioni fortemente correlate, non una dimostrazione d'indipendenza.

Spettri dei ranghi continui, anisotropia XY, correlazioni dei controlli e
tempo CPU di generazione saranno diagnostici, senza soglie di adozione.
Il costo di generazione esclude scrittura dei file; è informazione locale,
non un benchmark del frame né una prova di beneficio sul renderer.

Un fallimento viene conservato e riportato; nessuna soglia viene cambiata
in risposta ai dati. PASS significa soltanto superamento di questi gate per
questa tabella, questi seed e questa configurazione. Non certifica qualità
renderizzata, SOTA, GPU, energia, nuovi campionamenti runtime o convergenza
illimitata. La periodicità esatta di 16 frame è già nota e resta un limite
se il runtime ripete la stessa tabella senza ulteriori cambiamenti.
