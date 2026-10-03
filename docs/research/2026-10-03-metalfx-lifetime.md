# F8.4 — versioni MetalFX, lifetime e alternativa denoised

**Aggiornamento successivo:** il motore adotta [worker isolati](../F8_METALFX_LIFETIME.md)
per recuperare le risorse del framework al rilascio. Segue l’indagine che ha
motivato la soluzione; i suoi risultati negativi sull’SDK restano validi.

Indagine sul M5 Max, 2026-10-03. Base del motore: `862032e`.
**Esito iniziale dei percorsi in-process: nessuna correzione verificata del runtime.** Lo scaler
standard perde istanze e risorse; l'alternativa denoised libera le istanze ma
ha un'altra perdita CPU. Lo spike non è adottato e F8.4 resta aperta.

## Versione di compilazione e versione eseguita

Host: macOS 27.2, build `26B5091g`; Xcode 27.0 `27A266a`; Apple clang
21.0.0 `clang-2100.3.34.2`. Il framework caricato dal sistema è MetalFX
**40.9** (`/System/Library/Frameworks/MetalFX.framework`). Sono stati confrontati
SDK **26.5** (Command Line Tools) e **27.0** (Xcode), verificando anche
`LC_BUILD_VERSION` con `vtool`; deployment target 26.0 in entrambi.

Sono compilazioni diverse sullo **stesso runtime**, non prove su macOS 26.5.
Gli SDK forniscono header e stub di collegamento: cambiare SDK non sostituisce
il framework nel sistema. Anche le API Metal 3 e Metal 4 sono entrate nello
stesso backend temporale BBR, rispettivamente `_MFXTemporalScalingEffectBBR`
e `_M4FXTemporalScalingEffectBBR` (nomi osservati, mai usati per selezionare
comportamenti nel motore).

`softwareupdate --list` non offre aggiornamenti sul canale configurato.
Non sono stati modificati Xcode selezionato, canale beta o sistema operativo.
Le [note Apple di macOS 27.2](https://developer.apple.com/documentation/macos-release-notes/macos-27_2-release-notes)
consultate non documentano una correzione di questo specifico problema.
Non sono state provate altre versioni del **runtime macOS**.

## Prove indipendenti e impatto

[Runner](../../bench/f8_spike/run_lifetime_matrix.py) e
[riproduttore](../../bench/f8_spike/metalfx_lifetime_matrix.mm): creazione e
rilascio sequenziali, pool svuotato a ogni iterazione, controllo NSObject
per verificare il monitor weak, campionamento delle allocazioni del device.
Nessun render graph, texture applicativa o comando GPU. Il runner usa un
secondo processo senza monitor weak per il controllo `leaks`; conserva log,
exit, SDK, versione Mach-O, hash della sorgente e risultato di entrambi.

A 640×360, otto creazioni/rilasci del temporale standard lasciano **otto
scaler vivi**, sia con SDK 26.5 sia con SDK 27.0, tramite entrambe le API.
La differenza `currentAllocatedSize` arriva a **167.133.184 byte (159,39 MiB)**;
la sequenza cresce con le istanze. Una prova separata di quattro istanze
1920×1080 raggiunge **694.435.840 byte (662,27 MiB)**.

Questa misura include allocazioni/pool del driver: non è RSS, non è il
conteggio dei byte malloc e non va sommata direttamente a quest'ultimo.
L'aumento ripetuto e i target weak vivi confutano, nel campione osservato,
l'ipotesi di una singola cache limitata. Non sono byte persi a ogni frame:
il trigger è la **creazione e successivo rilascio dello scaler**, per esempio
quando cambia la dimensione di output. La DRS su backing fisso non implica
una nuova istanza a ogni cambio della regione attiva.

I circa **0,3 MB** del rapporto precedente erano soltanto quanto classificato
da `leaks` nella riduzione minima, non una misura completa delle risorse
trattenute. Senza observer, quattro istanze hanno prodotto 1.217.440 byte
malloc classificati come leak e il ciclo scaler ↔ filtro BBR. Alcune scansioni
restituiscono invece zero leak pur con target weak vivi e crescita del device:
la scansione conservativa non costituisce una prova di distruzione. I report
con esito zero sono conservati insieme a quelli che mostrano il ciclo.

Non risolvono il problema: creazione asincrona, esclusione della regione
dinamica/reactive mask, esposizione automatica del framework, `reset = true`,
dimensioni più piccole o rimozione dell'observer weak. La prova senza observer
mantiene la stessa crescita delle allocazioni. Il precedente drain di 30
secondi lasciava ancora vivo lo scaler.

## Alternativa temporale con denoising

Il controllo spaziale Metal 4 libera le istanze e supera `leaks`, ma non
implementa la ricostruzione temporale richiesta da F8.4.

Il temporale denoised Metal 3/4 libera gli scaler e le allocazioni GPU si
stabilizzano; **perde però 640 byte CPU per istanza**. La matrice da otto
istanze rileva 8 allocazioni/5.120 byte, mentre il test precedente da quattro
istanze rilevava 4 allocazioni/2.560 byte. Il difetto è riprodotto usando
soltanto il framework. Il semplice esito weak = 0 non chiude quindi il gate.

Il [probe di encoding](../../bench/f8_spike/denoised_encode.mm) esegue un
upscale 320×180 → 640×360 con tutte le guide e API/shader validation: termina
correttamente e il target weak viene distrutto. La maschera di bypass vale
**1**, come specificato nella
[documentazione Apple](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/denoisestrengthmasktexture).
Il controllo negativo di dimensioni produce prima dell'encoding del denoiser
l'assert `Color texture width mismatch from descriptor` (SIGABRT atteso).

L'API denoised richiede texture delle dimensioni fissate nel descrittore e
non espone `inputContentWidth/Height` del temporale standard. Quindi non
sostituisce direttamente il nostro percorso con backing fisso e regione DRS.
Una nuova integrazione dovrebbe gestire copie/crop, istanze per dimensione,
compilazione, invalidazioni e transizioni: queste capacità non sono dichiarate
implementate dal probe o dallo spike.

È stata provata anche una
[patch sperimentale nel motore](../../bench/f8_spike/denoised_native_spike.patch),
poi rimossa dal codice attivo. Usa creazione sui worker di PipelineCache,
accessi/fence del grafo, guide F7 e maschera 1; supporta esplicitamente soltanto
input nativo fisso. La smoke Sponza 640×360 sotto API/shader validation passa.
Nel motore il controllo `leaks` rileva ancora 640 byte attribuiti
all'inizializzazione dell'effetto denoised.

Confronto **esplorativo**, 180 frame Sponza, output 320×180, input 100%, camera
scriptata, riferimento radiometrico 4× per asse, controlli senza storia
corrispondenti a ciascun backend:

| Backend | PSNR medio sRGB | RMSE lineare p95 | Flicker residuo medio |
|---|---:|---:|---:|
| Temporale standard | 30,0448 dB | 0,022757 | 0,007460 |
| Denoised, bypass denoise | 29,0389 dB | 0,026914 | 0,008164 |

Entrambi passano le soglie esplorative riusate da F8 con i rispettivi controlli;
il denoised presenta più perdita di dettaglio nelle contact sheet. Questi
180 frame a scala 1 non sono il corpus di accettazione F8 da 480 frame a 0,75,
non certificano DRS e non dimostrano un vantaggio su altri contenuti. Non sono
misure prestazionali: la cattura PNG altera i tempi del frame.

Sono conservati anche due rapporti preliminari non validi per l'adozione:
il runner rigettava un messaggio INFO di archive miss contenente `error:`
(kernel nuovo assente nell'archivio); la ripetizione usa l'opzione pubblica
`--no-pipeline-archive`. Il primo confronto standard mancava del proprio
controllo senza storia; il successivo confronto usa il controllo corrispondente.
Nessuna soglia o immagine di riferimento è stata cambiata per assorbire errori.

## Decisione dello spike e prossimo controllo SDK (storico)

**Non adottare il denoised come workaround F8.4:** non supera il gate memoria,
non è equivalente per la DRS e nella prova eseguita riduce il dettaglio.
La patch è un artefatto di ricerca, non un backend supportato del prodotto.
Il codice attivo del renderer è stato ripristinato alla baseline, ricompilato
e ricontrollato. Native resta il default; il temporale standard è opt-in con
il residuo documentato. Nessuna OPT o fase successiva è stata attivata.

Per provare la regressione fra versioni serve un **diverso runtime macOS
fisico** compatibile con questo M5 Max, oppure un aggiornamento del framework
fornito da Apple. Non basta un altro SDK. La matrice deve prima provare la
distruzione delle istanze e l'assenza di crescita/leak; poi vanno ripetuti i
controlli del motore con resize, viste, DRS e qualità. Non sono stati usati
ivar privati, doppi release, spoofing del device o cache per nascondere il bug.
Nessuna segnalazione è stata inviata ad Apple.

Dati e comandi: `build/metalfx-investigation/`; sintesi versionata:
[risultati](../results/MetalFX-lifetime-M5Max-2026-10-03.json).
