# F12 — confronto congelato receiver corretto e griglie DDGI

## Esito sulla Cornell statica

La ricostruzione del receiver dalla primitive elimina il nero di cache/ReSTIR.
Con le stesse sei ROI e soglie preregistrate, passano **ReSTIR8×4×8,
DDGI16×8×16 e DDGI16×16×16**. Nessun nuovo preset è stato selezionato dopo aver
visto questi risultati. Fra le sole griglie DDGI esaminate,16×8×16 dà l'errore
massimo più basso e costa meno di16³: è un candidato di qualità per questa
Cornell, non un preset universale o già adottato per prestazioni.

Il protocollo aggiuntivo `tools/testdata/f12/grid-sweep-protocol-v1.json`
è stato committato in cf44d22 prima di aprire i pixelv3. Mantiene il protocollo
953e23e: media32capture256..504,128×72, seed1001, sei ROI geometriche erose2px,
NRMSE_RGB≤30%, |bias luminanza|≤20%, nessun fit/esposizione/frame selection.
Tutti i sei run sono sul commit68b635c836b8b13bf094a2c23eb69d0f0775ba08, stesso
snapshot SHA256`9ea09eae5e426c449791b43c38bc3521a223e0708442861f3a4e923e1f5d237f`.
Il riferimento indipendente65536spp×2seed non è stato rigenerato né modificato.

| Modalità/griglia | Peggiore NRMSE | Peggiore bias assoluto | Sei ROI |
|---|---:|---:|---|
| DDGI8×4×8 |39.26%|27.16%|FAIL|
| Cache8×4×8 |38.15%|4.13%|FAIL|
| ReSTIR8×4×8 |29.34%|11.15%|PASS|
| DDGI8×8×8 |48.50%|36.66%|FAIL|
| DDGI16×8×16 |23.31%|5.31%|PASS|
| DDGI16×16×16 |28.91%|10.32%|PASS|

Il FAIL cache è sul box corto, con media relativamente vicina ma forte
variabilità temporale: RMS temporale dell'interno97.37% del RMSreference,
contro51.33% ReSTIR e0.77% DDGI di base. Questo è coerente con una componente
importante di rumore; non basta a provare che ogni errore cache sia solo varianza.
Il gate raw rimane FAIL, senza estendere il numero di frame dopo aver visto i dati.
F13 deve verificare il proprio denoising separatamente. ReSTIR passa con margine
ridotto sul box corto e mantiene il bias di supporto già dichiarato dal contratto.

Dettaglio dei due preset DDGI che passano (NRMSE / bias):

| ROI |16×8×16|16×16×16|
|---|---:|---:|
| floor |6.84% /+5.31%|5.28% /+4.15%|
| back |3.89% /+2.50%|4.19% /+3.59%|
| red_wall |3.86% /−0.63%|5.91% /+2.69%|
| green_wall |3.72% /−0.84%|4.23% /+0.81%|
| short_box |23.31% /+3.75%|28.91% /−10.32%|
| tall_box |5.86% /+2.15%|4.57% /+4.33%|

Più probe non garantiscono un miglioramento monotono: cambiano collocazione,
interpolazione e relocation.8³ peggiora, e16³ non domina16×8×16. I dati non
isolano ancora quanto ciascun meccanismo contribuisca; non viene introdotto
un refactor dell'algoritmo per inseguire questo singolo scenario.

## Costi effettivamente registrati

**Singolo run con API e shader validation**,256frame misurati,512totali.
Questi valori affiancano il confronto di qualità; non costituiscono adozione
prestazionale. Il costo è del frame completo con diagnostica/capture, non GI
isolata. Ogni pass nel report ha un solo campione: non si sommano mediane dei
pass e non si chiamano quei dati una distribuzione steady-state.

| Modalità/griglia |GPUframe p50 / p95 ms|Risorse engine MiB finali|Raggi probe/frame|
|---|---:|---:|---:|
| DDGI8×4×8 |2.542 /4.226|436.03|16.384|
| Cache8×4×8 |2.636 /4.383|436.03|16.384|
| ReSTIR8×4×8 |3.026 /4.729|436.03|16.384|
| DDGI8×8×8 |2.619 /3.980|445.04|32.768|
| DDGI16×8×16 |3.785 /6.088|499.13|131.072|
| DDGI16×16×16 |4.753 /7.031|571.24|262.144|

Il confronto16×8×16 contro16³ evita circa72.11MiB di risorse contabilizzate e
0.968ms di p50 osservato in questi run, con qualità migliore sul criterio
peggiore. L'allocator del device si muove per blocchi:683.95MiB contro812.02MiB;
questo dato è separato dalle risorse engine. Ogni run riporta2allocazioni GPU:
questa batteria con capture/export non certifica O7. Non è stata attribuita la
loro causa qui. Misure senza validazione/diagnostica, macchina quieta e tre
repliche restano necessarie prima di qualsiasi scelta prestazionale.

## Perché il fit conservativo spreca densità potenziale

Dal solo snapshot è ricostruito il volume delle sfere usato dal fit:
min(−4.828,−2.828,−4.828), max(4.828,6.828,2.828), poi padding0.25m.
La stanza è4×4×4m. Le sfere delle grandi pareti planari allargano il volume
anche lungo il loro spessore nullo. Con il fix b9174f2, i counts sono applicati
prima dello spacing, perciò tutti i preset coprono lo stesso volume.

| Griglia |Spacing m|Probe inizialmente nella stanza /totali|Di queste dentro box|
|---|---|---:|---:|
|8×4×8|1.451×3.386×1.165|16 /256|2|
|8×8×8|1.451×1.451×1.165|16 /512|1|
|16×8×16|0.677×1.451×0.544|96 /2.048|1|
|16×16×16|0.677×0.677×0.544|288 /4.096|9|

Sono conteggi CPU **prima** di relocation/classificazione, non un readback
GPUné il numero delle probe attive a regime. Mostrano che il fit conservativo
può spendere gran parte del budget fuori dal volume di interesse e che8³
raddoppia il lavoro senza aumentare il conteggio iniziale interno. Non provano
da soli la causa di tutti gli errori misurati. Un fit più stretto è un'opportunità
successiva da misurare sul corpus, non una modifica fatta in questo confronto.

## Limiti e riproduzione

Il PASS indicato riguarda soltanto le sei ROI statiche. La ROI leak contiene
1pixel, sotto il minimo16: resta NONCERTIFICABILE. Recovery, thin walls,
probe inside wall, altre scene/seed, denoising, dispositivi fisiciApple9 e
prestazioni reali non vengono chiusi da questa prova.

`tools/f12_grid_compare.py` verifica il set esatto dei sei casi, la configurazione
CLI, il volume fisico/camera tramite hash, il budget raggi e la provenienza dei
costi; richiama la stessa funzione di qualità del confronto originale.
Il refactor è stato verificato rieseguendo v2: JSONidentico prima/dopo.
Sono stati provati anche i rifiuti di griglia errata e spacing non autorizzato
per il protocollo; nessuna GPU eseguita da questo agente.

Dati: `docs/results/F12-bounded-grid-M5Max-2026-10-07.json`.
Raw: worktreef9-proxy `build/f12-oracle-v1/grid-quality-v3/`.
Mosaico `comparison-crop.png`: riga1 reference /DDGIbase /cache /ReSTIR;
riga2 DDGI8³ /DDGI16×8×16 /DDGI16³. Ritaglio e scala6× identici, solo per
illustrazione; le metriche usano sempre i PFM completi lineari.
