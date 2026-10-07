# STBN consumato: secondo esperimento CPU, blocchi 0–3

## Protocollo preregistrato prima dei nuovi dati

**Stato iniziale: PREREGISTERED.** Il precedente esperimento sui ranghi originali
rimane **FAIL** (`67641c9`, `STBN-cpu-spectrum-M5Max-2026-10-07.json`) e non viene
sostituito o riclassificato. Questo secondo esperimento è richiesto per verificare
la trasformazione effettivamente applicata ai campioni.

Sorgente congelato: `12adf1218234e80ff8c67f6b3eac78a7cda71f0c`, versione 2,
`StbnMask::sample` con rotazione Cranley–Patterson per dimensione **e blocco**.
La differenza rispetto a `8e533bf` riguarda `sample` e la versione; il corpo di
`generateStbn` è identico. Riutilizziamo gli otto file di ranghi originali,
verificati con SHA-256 contro il risultato S1. Nessuna nuova maschera, nessun
nuovo seed, nessun tuning del generatore.

Per ogni seed 0–7 si considerano i blocchi 0,1,2,3: frame 0–15,16–31,32–47,48–63,
64 dimensioni e tutti i pixel 8×8. La formula è riprodotta con **operazioni
float32** in Python e confrontata bit per bit con un eseguibile CPU che chiama
il vero `StbnMask::sample` del sorgente congelato. Qualsiasi differenza blocca
la conclusione: non si allarga una tolleranza per far combaciare le due versioni.

Le metriche e tutte le soglie di S1 sono invariate, applicate **separatamente a
ciascun blocco**: cinque livelli CDF, FFT XY bassa frequenza `0<kx²+ky²≤2`, FFT Z
`|kt|=1` su 16 campioni, media del rapporto ≤0,75 e almeno 7/8 seed migliori di
ciascun controllo; Pearson tra dimensioni p95≤0,10 e massimo≤0,25 per seed/blocco.
Il PASS complessivo richiede il PASS di tutti e quattro i blocchi.

I controlli rimangono permutazione uniforme e IID uniforme, ma hanno stream
indipendenti per blocco (`SeedSequence([seed,1398030926,dimension,control,block])`).
Uniformità significa campioni in [0,1), `floor(sample*1024)` permutazione esatta
per dimensione/blocco e 32 campioni per bin dei 32 bin. Questo è l'equivalente
operativo del gate sui ranghi dopo una rotazione uniforme. Rilassamento e
provenienza delle maschere provengono da S1, senza rigenerarle.

La sequenza intera di 64 frame viene anche analizzata a `|k|=1` come diagnostica:
è una scala temporale diversa e non viene nascosta dentro la media dei gate
sui blocchi. Non certifica da sola l'assenza di artefatti alle frontiere.
Il risultato riguarda valori CPU effettivamente consumati; nessuna GPU,
prestazione del frame, immagine renderizzata o claim SOTA. Soglie e seed restano
immutati dopo l'esecuzione; un fallimento viene conservato.

Protocollo macchina: `tools/testdata/stbn/protocol-consumer-v2.json`.
