# F12 — segnale filtrato reale, oracolo thin-wall e stato emissive-step

## Cornell filtrata: sei ROI PASS nei quattro casi congelati

Il nuovo captureID8 legge il vero E selezionato da F13custom, moltiplicato una
volta per albedo×(1−metallic)/pi, prima di AO artistico/exposure/tonemap.
Non è un'immagine filtrata offline. Protocollo `filtered-protocol-v1.json`
committato prima di leggere i candidati: stesse sei ROI Cornell, reference
65536spp×2seed, media32capture256..504, seed1001,128×72, gate20%bias/30%NRMSE.

| Caso |Peggiore NRMSE|Peggiore bias assoluto|Sei ROI|
|---|---:|---:|---|
|Cache/custom8×4×8|29.48%|13.29%|PASS|
|Cache/custom16×8×16|27.99%|7.70%|PASS|
|DDGI/custom16×8×16|24.19%|5.12%|PASS|
|ReSTIR/custom8×4×8|28.83%|13.21%|PASS|

Il risultato rawcache precedente **resta FAIL** (38.15%NRMSE): è un segnale
diverso. Cache/custombase e ReSTIRcustom hanno poco margine sul limite30%.
Il confronto non certifica altre scene, seed o la recovery temporale del filtro.
Anche qui la ROI leak Cornell ha solo1pixel e rimane insufficiente.

I p50GPU registrati sono rispettivamente3.898/5.051/4.915/4.275ms, da singoli
run con API+shader validation e capture. Ogni pass ha1campione e i run riportano
2allocazioni GPU: nessuna adozione prestazionale/O7. Tutti i dettagli sorgente,
configurazione, frame e regione sono nel JSON, senza sostituire i report raw.

## Snapshot reali emissive-step: transizione provata

Gli snapshot runtime255 e256 passano `f12_step_snapshot_check.py`:
assetPLY/PFM identici, stessa geometria/camera/material response, unica lampada
Le12→6 e distribuzioni degli emettitori dimezzate. Sono ignorati solo frame,
revisioni e generation diagnostici, mai i campi fisici.

- Frame255 SHA256`9ee1d1a25a3c7c3eb83b65c8382912d12350af27d7cf19f53d59a77329858261`.
- Frame256 SHA256`ba48de1ae9d4a00d017ac87067dbd0787766883341c3b8f6d21f45da3d749d30`.

Il riferimento dopo il cambio è quindi esattamente0.5×l'oracolo Cornell
indipendente. Questo verifica lo stato, **non ancora la recovery delle immagini**.
Restano i checkpoint0/1/4/8/16/32/64/128 e il gate F13history≤4 separato.

## Thin-wall: reference indipendente convergente, qualità motore non ancora confrontata

La fixture reale256×144 ha parete chiusa10mm, lampada nel comparto sinistro,
zeroaltreluci/sky. SnapshotSHA256:
`7b20c70095d3b3cfb8cb430c2f51f18219364ac4163b874e2b4870d4461928ac`.
Le maschere geometriche congelate prima di qualsiasi pixel candidato contengono
**849bright e830dark**; sono erose2px anche sui confini fra superfici.

Mitsuba3.9.1scalar_rgb,2thread CPU, full unlimited meno depth2, Float32 firmato.
Le1679intersezioni ROI coincidono con il ray/triangle oracle indipendente:
0mismatch, erroreWORLD≤3.33µm, UV≤1.01e−6, normali≤2.99e−8. Il plugin texture e
la specializzazione costante nativa producono gli stessi pixel a64spp(maxdiff0).

Il riferimento converge a16384spp per ciascuno dei seed1234/5678:

- Bright: differenzaNRMSE1.416%, sotto2%.
- Dark: differenzaRMS/meanBright0.0942%, sotto1%.
- Dark: differenza della media/meanBright0.00211%, sotto0.2%.

Il checkpoint4096spp era FAIL bright2.798% e viene conservato. La reference
finale è la media dei due seed16384spp. Dark non è assunta nera: la soluzione
indipendente include i percorsi fisici attorno all'apertura della parete.
Nessun pixel della futura qualità thin-wall del motore è stato letto qui.

Il negativo fisico `--debug-gi-no-visibility` è stato consegnato come sorgente
in61fa29e: disabilita solo il peso di visibilità basato sui momenti. Il normale
checker deve restare PASS; deve fallire la metrica di leak indipendente.
Compilazione/GPU e confronto positivo/negativo sono ancora compiti del root.

## Evidenza e limiti

Dati committati:
`docs/results/F12-filtered-Cornell-M5Max-2026-10-07.json` e
`docs/results/F12-thin-oracle-step-state-M5Max-2026-10-07.json`.

Raw nel worktreef9-proxy:
`build/f12-oracle-v1/filtered-quality-v1/`,
`build/f12-step-reference-real-v1/`, `build/f12-thin-reference-v1/`.
Mosaico filtrato: reference/cachebase/cachedense;DDGIdense/ReSTIRbase, scala6×
e ritaglio identici, solo illustrativo. Le metriche restano sui PFM lineari.

Nuovo script CPU `tools/f12_thin_reference.py` registra ogni checkpoint e il
fallimento delle condizioni, senza adattare soglie o leggere immagini motore.
Non sono stati eseguiti build del motore né workload GPU da questo agente.
Il PASS statico non chiude F12/F13, leakage, recovery o certificazione hardware.
