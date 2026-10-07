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

## Esito S2: PASS del protocollo sui valori consumati

Preregistrazione `ddbe3d5`; strumenti congelati in `d81916d` prima del dump/FFT. Il checker termina con **exit 0**. Non sono stati cambiati seed, soglie o generatore. Il risultato S1 e il suo SHA-256 sono rimasti invariati: **raw S1 = FAIL**, **consumo S2 = PASS**.

Confrontati **2.097.152 valori float32** Python contro il vero C++ `StbnMask::sample` del commit congelato: **zero differenze bit per bit**. Tutti i campioni sono in [0,1); le 2.048 coppie dimensione/blocco/seed hanno `floor(sample*1024)` permutazione esatta e istogramma a 32 bin uniforme. Le maschere sono state riutilizzate, non rigenerate.

| Blocco | Gate spettrali | Peggior rapporto XY | Peggior rapporto Z | Peggior p95 Pearson | Max Pearson |
|---|---:|---:|---:|---:|---:|
| 0 | 20/20 PASS | 0.343847 | 0.586541 | 0.074621 | 0.159609 |
| 1 | 20/20 PASS | 0.341404 | 0.582635 | 0.074514 | 0.155257 |
| 2 | 20/20 PASS | 0.339542 | 0.586505 | 0.078131 | 0.158885 |
| 3 | 20/20 PASS | 0.339222 | 0.579138 | 0.075483 | 0.152499 |

I **80 gate spettrali** passano: per ogni combinazione CDF/asse/controllo STBN migliora in **8/8 seed**. Tutti i gate di decorrelazione passano. Anche il controllo diagnostico sulla sequenza completa di 64 frame ha rapporti medi rispetto ai due controlli compresi fra 0,213821 e 0,522468 a `|k|=1`; questo dato resta fuori dai gate preregistrati sui blocchi. I quattro blocchi hanno hash distinti per ciascuno degli otto seed.

**Interpretazione limitata:** la rotazione dei valori mantiene una soppressione misurata delle basse frequenze e riduce la correlazione scalare rispetto ai ranghi grezzi, entro questo protocollo. Parte essenziale della differenza raw→consumo proviene dalla rotazione **per dimensione già esistente**: il blocco 0 ha quella stessa trasformazione. Non attribuiamo quindi il miglioramento alla sola nuova rotazione per blocco. Non sono misurate indipendenza congiunta completa, qualità del ReSTIR, spettro GPU, denoising, convergenza illimitata o guadagno sul frame.

Risultato completo: `docs/results/STBN-consumer-spectrum-M5Max-2026-10-07.json`. I campi `generation_ms` provengono dal precedente S1 e non rappresentano una nuova misura del costo di `sample`; qui non è stata eseguita generazione. Hardware/compilatore/runtime sono gli stessi del report S1.

Comandi eseguiti nel solo build privato CPU:

```sh
cmake -S build/stbn-consumer-v2/tool -B build/stbn-consumer-v2/tool-build -G Ninja
cmake --build build/stbn-consumer-v2/tool-build -j2
./build/stbn-consumer-v2/tool-build/stbn_consumer_generate \
  build/stbn-spectral/ranks-seeds-0-7 build/stbn-consumer-v2/samples
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/danielsan/Documents/phosphor/build/quality-venv/bin/python tools/stbn_consumer_check.py \
  --ranks build/stbn-spectral/ranks-seeds-0-7 \
  --consumer build/stbn-consumer-v2/samples \
  --snapshot build/stbn-consumer-v2/snapshot \
  --output docs/results/STBN-consumer-spectrum-M5Max-2026-10-07.json
```

Lo snapshot contiene `stochastic_sampling.h/.cpp` e `core/types.h` estratti con `git show 12adf12:<percorso>`: non sono state modificate le copie del generatore di questo worktree o delle altre chat. Il corpo di `generateStbn` è confrontato byte per byte con S1 prima di consentire il riuso dei ranghi.
