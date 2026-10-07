# F12 — zero GI da receiver dietro alla geometria

Il difetto cache/ReSTIR è riprodotto su capturev2, con il fix970d7ce dell'età
attivo. Gli stessi gate preregistrati restano FAIL: floor/red/green sono zero;
tall_box conserva bias−78.93% in ReSTIR. DDGI e cache sono identici av1.

I capture `shadow-position`/`shadow-normal` del frame511 sono confrontati con
intersezioni indipendenti in doppia precisione della stessa PLY. Le normali
sono corrette; i punti floor e pareti sono **165–227µm dietro al proprio piano**.
L'errore WORLD massimo è0.913mm. Sul backplane è inferiore a0.6µm. Il punto
si ottieneva con `inverseViewProjection * depth`, senza usare il piano della
primitive del V-buffer già disponibile. Il depth rasterizzato non garantisce
un punto entro l'errore di arrotondamento della vera geometria RT.

W&B aggiunge15.26µm alla coordinata prossima a zero e pochi ULP sulle pareti:
non può recuperare un punto già centinaia di micrometri dentro alla geometria.
I raggi cosine quindi incontrano subito il retro della propria faccia;
gi_candidates li conta correttamente come proposte zero e l'immagine resta nera.
Aumentare indiscriminatamente il bias sposterebbe il problema sui dettagli fini.

## Correzione isolata

`tools/patches/f12-exact-primitive-receiver.patch` cambia solo
`shaders/shadows.metal`. Usa `GPUShadowParams.viewProjection`, i tre vertici
WORLD del V-buffer e il helper F7 `visibilityBarycentrics` già incluso;
ricostruisce WORLD con le stesse baricentriche del material resolve. Depth
rimane disponibile per verifica/riproiezione; non è la posizione di partenza RT.
La soluzione gestisce anche primitive che attraversano il near plane secondo
il contratto già verificato del helper omogeneo, senza nuova ABI o bias.

Prova CPU: `tools/f12_receiver_barycentric.cpp` compila direttamente il helper
congelato sotto esame. `tools/f12_receiver_probe.py` confronta il risultato
float32 con il ray/triangle oracle double, applica le costanti W&B invariate e
traccia16 direzioni cosine stratificate indipendenti per ciascuno dei781 pixel.

| ROI | Raggi | Self-hit prima | Self-hit dopo |
|---|---:|---:|---:|
| floor |912|912|0|
| back |4608|0|0|
| red_wall |2784|2784|0|
| green_wall |2592|2592|0|
| short_box |848|0|0|
| tall_box |752|560|0|

Totale:6848/12496 self-hit prima,0 dopo. Errore sul piano dopo≤0.326µm;
WORLD contro oracle≤1.21µm. Sono prove CPU sui dati GPU catturati prima della
patch, **non** un'esecuzione Metal della correzione. Root deve compilare MSL,
ricatturare guide e GI e riapplicare gli stessi gate; nessuna qualità dichiarata
PASS in anticipo. Le immagini zero/doppia energia restano negativi del checker.

L'overflow indexed usa `in.worldPos` interpolata direttamente dai vertici,
non questo percorso depth-unprojection. Non è modificato. Una capture forzata
dell'overflow va verificata contro lo stesso oracle prima della chiusura,
soprattutto per superfici ruotate e world coordinates grandi.

DDGI short_box rimane un problema separato di qualità:−27.17%bias/39.27%NRMSE,
senza self-hit della camera; nessuna prova che la patch lo risolva. Il volume
attuale deriva dalle sfere conservative degli oggetti e ha pochi layer utili
interni; occorre misurare discretizzazione/grid/relocation, senza allargare i
limiti di accettazione o attribuire quel bias al difetto appena isolato.

Evidenza numerica: `docs/results/F12-receiver-self-hit-M5Max-2026-10-07.json`.
Raw nel worktreef9-proxy: `build/f12-oracle-v1/receiver-barycentric-v2/`,
`guide-validation-v2.json`, `quality-v2/`. Nessuna GPU usata da questo agente.
