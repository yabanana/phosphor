# F8.4 — gestione del lifetime MetalFX completata sul M5

**DEVELOPMENT_ACCEPTED, 2026-10-03**, PR #15. Il motore usa MetalFX temporale
standard in worker isolati, con memoria limitata al loro ciclo di vita.
**Il ciclo di riferimenti interno di MetalFX 40.9 non è stato corretto:** il
riproduttore diretto resta negativo. La soluzione del motore termina e recupera
il contesto del framework al rilascio, invece di lasciare accumulare gli scaler
nel processo del renderer. Nessuna OPT o fase successiva è stata attivata.

## Perché questa soluzione

La [prima indagine](research/2026-10-03-metalfx-lifetime.md) confronta SDK 26.5/27.0,
API Metal 3/4 e un'alternativa denoised. Anche un
[programma Swift con ARC](../bench/f8_spike/metalfx_arc_lifetime.swift), senza
wrapper Phosphor e con il main dispatch attivo, lascia vivi otto scaler su otto.
Il denoised perde meno memoria ma non supera il controllo completo e cambia
qualità/contratto della DRS. Non è stato adottato.

Il worker conserva **lo stesso algoritmo, formati, jitter, motion, esposizione,
reactive mask e input dinamico** del temporale standard. Nel confronto Sponza
641×361, input 75%, frame 99, direct e isolated sono pixel-identici:
**0/231.401 pixel diversi**, tolleranza zero. Non è una sostituzione con un
upscaler spaziale o con un nuovo algoritmo di qualità non verificata.

## Ownership e sincronizzazione

- Un worker persistente per vista e dimensione di output. La DRS cambia la
  regione attiva dello stesso scaler. Camera cut e cambio scena resettano la
  storia della vista. Il resize di output prepara un nuovo worker e ritira
  quello precedente; durante l'avvio il renderer mantiene il fallback nativo.
- Socketpair privata e file condiviso anonimo dopo `unlink`, ereditati solo
  dal figlio tramite `posix_spawn`. Non c'è un servizio di rete/launchd e non
  vengono modificati OS, canali di aggiornamento o API/ivar privati.
- Tre slot distinti nel trasferimento. Le copie di colore/depth/motion/reactive/
  exposure e HDR ricostruito sono GPU; la CPU trasporta solo parametri e stato.
  I tre slot non sono sostituibili con uno senza una nuova prova delle
  dipendenze fra frame.
- Il grafo divide il pass External in producer e consumer: segnala input-ready
  dopo il producer, attende output-ready prima del consumer. Il broker attende
  e comunica fuori dal render thread. Il job diventa eseguibile solo dopo la
  pubblicazione della submission, evitando timeout su comandi ancora in registrazione.
- GpuMemory possiede i buffer/texture di entrambi i processi. Il buffer di
  scambio mantiene mapping, eventi e worker fino all'ultimo lettore GPU. Il
  retirement e il recupero del processo avvengono fuori dal percorso caldo.
  Un contesto worker headless non alloca inutilmente ring di upload/presentazione.
- Massimo otto processi attivi/in retirement. Se la capacità è piena, l'avvio
  riprova nelle frame successive senza bloccare render thread o compile queue.
  I descriptor non correnti vengono cancellati. Tutti i figli devono essere recuperati.
- EOF, risposta errata, crash o timeout con completamento GPU incerto fanno
  terminare il renderer con errore. Non si azzera/riusa un output che il figlio
  potrebbe ancora scrivere. La distruzione del processo elimina le sue attese
  GPU; i descriptor close-on-exec fanno arrivare EOF agli altri worker.

La scelta vale per il backend macOS corrente. L'API diretta resta disponibile
con `--metalfx-mode direct` **per diagnostica e confronto**, con il difetto noto
sul runtime verificato. Il default generale resta `--upscaler native`;
`--upscaler temporal` seleziona l'isolamento. Non occorre cambiare versione di macOS.

## Verifiche

- 437 test portabili, 1.773.691 asserzioni; build Debug/Release; CI Linux con
  controlli sintattici Metal host e CI macOS con app/shader/archivio pipeline.
- Suite F7/F8 funzionale: guide, motion, età delle pose, esposizione, curve,
  negativi, dimensioni dispari, 1/2/3 frame in volo, fallback Apple9 sul M5; quattro viste con EDR e DRS sotto validation.
- Corpus da 480 frame Torus/Sponza, tre camera cut, varianti tile/adaptive,
  controlli senza storia e negativi ritardati. Clip aggiuntive esposizione e
  DRS/due viste passano senza modificare soglie o riferimenti.
- Async compute ed encoding suddiviso passano sotto API/shader validation,
  incluso il guasto simulato del worker. Anche il percorso nativo supera
  nuovamente `tools/f6_check.sh --quick` (immagini, controlli, eventi, switch/resize).
- Protocollo/versione non valida respinti; 35 secondi di inattività non fanno
  uscire prematuramente il worker. La morte forzata del parent non lascia orfani.
- Crash e timeout intenzionali producono exit 1 entro il limite, senza fingere
  un frame riuscito. La terminazione fail-closed può non stampare il marker
  ordinario `EXIT`; il runner conserva il vero codice di processo.
- Test fisso: **20.000 frame misurati +240 warmup**, Sponza 1080p/input 75%,
  frame medio 2,1725 ms, p95 2,4572 ms, p99 2,7239 ms. Zero nuove allocazioni
  GPU del motore (parent e worker), zero ricostruzioni dei command buffer,
  zero errori, un worker creato/recuperato e zero byte condivisi alla chiusura.
- `leaks --atExit` del renderer con temporale isolato: **zero leak**. Il
  riproduttore SDK diretto rimane disponibile e fallisce: non è stato escluso
  o reinterpretato come dimostrazione della correzione interna del framework.

La certificazione fisica M3/T0 e la fotometria HDR restano quelle dichiarate
nella [policy hardware](plans/HARDWARE_VALIDATION.md). L'uscita storica F6 non
attribuita non viene retroattivamente spiegata da questo lavoro.

## Memoria con resize ripetuti

Stesso percorso, due viste, 240 frame e resize ogni 30 frame; pacing diagnostico
64 ms per permettere agli scaler di diventare attivi tra i resize. È un test
lifetime, **non una misura prestazionale**. I run rapidi precedenti, dominati dal
fallback durante l'avvio, sono conservati e non usati come prova sostitutiva.

| A fine misura, prima della chiusura | Direct | Isolated |
|---|---:|---:|
| Footprint fisico del parent | 7.440.994.504 B | 1.809.122.360 B |
| Footprint riportato dai worker attivi | — | 1.280.182.096 B |
| Allocazioni device del parent | 6.942.457.856 B | 1.943.322.624 B |
| Frame realmente temporali | 225/240 | 226/240 |

Isolated: **16 worker creati, 16 recuperati, massimo quattro vivi**; zero mapping
rimasti dopo teardown. Il campionamento usa `proc_pid_rusage`/footprint:
RSS da solo non misura correttamente la memoria GPU/IOKit. Le cifre di processi,
device, heap e mapping sono domini differenti; memoria condivisa e pool non
vanno sommati ciecamente come RAM fisica distinta.

## Costo sul frame

Release, M5 Max, macOS 27.2/MetalFX 40.9, 1920×1080, input 75%, 120 warmup
+600 frame, tre repliche in ordine ruotato, no UI/vsync/validation, offscreen.
La tabella usa la mediana dei tempi medi e il p95 della replica mediana.

| Carico | Direct medio | Isolated medio | Differenza | Isolated p95 |
|---|---:|---:|---:|---:|
| Sponza | 2,1503 ms | 2,1851 ms | +1,62% | 2,4530 ms |
| 1.024 luci | 12,0212 ms | 13,1646 ms | +9,51% | 14,3532 ms |
| Sponza, due viste alternate | 1,2743 ms | 1,9189 ms | +50,58% | 2,2582 ms |

La prima batteria dava +0,42%/+7,34%/+51,27%; entrambe sono conservate. Non
si seleziona soltanto il confronto più favorevole. Il costo multiview è reale,
anche se il tempo assoluto nel preset rimane basso. Sono throughput offscreen,
non FPS presentati. Il tempo CPU standard riguarda il render thread e non
misura il lavoro totale dei worker. Nessun miglioramento energetico è dichiarato.

Il report **schema 8** separa footprint/allocazioni parent-worker e mapping,
conta le allocazioni GpuMemory di entrambi e dichiara il tempo GPU come span
fra prima e ultima submission graphics: include l'intervallo IPC/esterno,
non rappresenta solo tempo di esecuzione GPU attiva. I costi di copie, memoria
e sincronizzazione sono parte della soluzione, non nascosti nelle statistiche.

## Riproduzione e limiti

```sh
mise exec -- python3 tools/metalfx_lifetime_check.py
mise exec -- python3 tools/metalfx_protocol_check.py
mise exec -- python3 tools/metalfx_isolation_bench.py
build/quality-venv/bin/python tools/f7_f8_check.py --build build/release --quality-only
build/quality-venv/bin/python tools/f7_f8_check.py --build build/release --context-quality
```

Un solo workload GPU coordinato alla volta: i suoi worker fanno parte della
prova, non sono benchmark indipendenti concorrenti. Le catture/readback,
le scansioni leak e il pacing diagnostico non valgono come tempi di performance.
Gli esperimenti e i report falliti precedenti restano conservati. Lo spike
denoised resta un artefatto storico non applicato. Nessuna patch privata del
framework o cache di scaler sopravvissuti è stata introdotta.

Dati compatti: [risultati](results/MetalFX-isolation-M5Max-2026-10-03.json).
Grezzi locali: `build/metalfx-recheck/`. Per tornare al percorso diretto su un
runtime futuro servono la matrice lifetime SDK e le prove del motore: non si
presume una correzione dalla sola versione dell'SDK di compilazione.
