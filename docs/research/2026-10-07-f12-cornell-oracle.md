# F12 — oracolo CPU Cornell e confronto congelato

## Esito

L'oracolo indipendente è validato per questa Cornell diffusa statica128×72;
la qualità F12 **non passa** il protocollo preregistrato. DDGI fallisce sul box
corto; cache e ReSTIR restituiscono zero su pavimento e pareti laterali. Nessuna
soglia è stata modificata per accettare i dati. F12 non è dichiarata completata.

Protocollo: `tools/testdata/f12/oracle-protocol-v1.json`, commit953e23e, scritto
prima di leggere qualunque PFM candidato. Codice candidato9215a03e; i capture
precedono il successivo fix dell'età della catena ReSTIR. Tutti i check GPU del
root erano PASS: non misuravano questi errori rispetto al trasporto indipendente.

## Oracolo e verifica del contratto

Mitsuba3.9.1 `scalar_rgb`, due thread CPU, nessuna GPU. Snapshot SHA256:
`9ea09eae5e426c449791b43c38bc3521a223e0708442861f3a4e923e1f5d237f`.
Metri, luce emessa Le=(12,12,12), lampada0.800000012m², un solo emettitore;
le due distribuzioni triangolari esportate non aggiungono luci duplicate.

La verifica indipendente dei9216 raggi primari ricostruisce camera/near clip,
triangoli PLY, trasformazioni WORLD, winding e UV. Le781 posizioni interne
coincidono entro1.65e-6m, UV4.25e-7, normali9.78e-8. Tre tie su spigoli della
stanza sono registrati ed esclusi dall'erosione2px già prevista. È verificata
la camera di questo snapshot, con jitter0; non ogni possibile camera/asset.

Otto casi texture2×2 verificano orientamentoUV, repeat negativo, bilinear,
conversione half dopo filtraggio e MASK dopo filtraggio: errore0. La
specializzazione Cornell dei materiali costanti nativi produce gli stessi
pixel del plugin generico a64spp (maxdiff0). Raddoppiare Le raddoppia il render
(maxdiff0); riflettanza zero elimina tutto l'indiretto (maxdiff0).

L'adapter consegnato non poteva caricare lo snapshot: `area`3.9.1 rifiuta
`twosided` e `sample_texture`. Inoltre il flag `is_spatially_varying=True`
selezionava la proposta nello spazioUV, inadatta alle mesh con UV sovrapposte.
La patch isolata `tools/patches/f12-reference-mitsuba391.patch` rimuove le
proprietà inesistenti, rifiuta emissioni two-sided non supportate e usa una
proposta uniforme sulla superficie anche quando Le/MASK è valutata con UV.
La valutazione texture resta identica. Il confronto64spp tra le due vecchie
proposte differiva di0.0376: è conservato, non prova da solo un bias Cornell.
Dopo il fix la proposta e il render coincidono con l'emettitore nativo.

La patch abilita anche `--max-depth -1`. Per definizione Mitsuba depth2 è
illuminazione diretta; il segnale è **full unlimited meno depth2**, Float32
lineare e firmato, senza clamp dell'errore o fit di esposizione. La differenza
indipendente fra seed1234/5678 converge sotto2% per ogni ROI solo a65536spp:

| ROI | Pixel | Differenza fra seed |
|---|---:|---:|
| floor |57|0.850%|
| back |288|0.926%|
| red_wall |174|0.864%|
| green_wall |162|0.872%|
| short_box |53|1.090%|
| tall_box |47|0.729%|

Checkpoint64/256/1024/4096/16384 e relativi FAIL sono conservati.
A4096spp il peggiore errore era5.45%; a16384 il box corto era2.26%.
Il riferimento adottato è la media dei due seed65536spp. Depth8 rispetto a
unlimited sottostima la luminanza delle ROI dell'1.01–2.38% (interno1.60%).
Le durate CPU non sono benchmark né una scelta prestazionale per il motore.

## Confronto con i capture del motore

Segnale controllato nel codice: `gi_reference_diffuse` applica
E_indirect×albedo×(1−metallic)/π, prima di AO artistico/esposizione/tonemap.
Si confronta la media dei32 capture256,264,…,504. Snapshot fisico e camera
di tutte le varianti hanno lo stesso hash. Nessun ridimensionamento, fit,
scelta di frame o maschera in base all'immagine. Gate per ogni ROI:
NRMSE_RGB≤30%, |bias luminanza|≤20%.

| ROI | DDGI NRMSE / bias | Cache NRMSE / bias | ReSTIR NRMSE / bias |
|---|---:|---:|---:|
| floor |12.89% /−2.69%|100% /−100%|100% /−100%|
| back |13.59% /+6.18%|16.79% /−0.78%|13.05% /−1.59%|
| red_wall |9.58% /−5.89%|100% /−100%|100% /−100%|
| green_wall |13.81% /+2.57%|100% /−100%|100% /−100%|
| short_box |**39.27% /−27.17%**|**38.15%** /−4.13%|29.38% /−10.16%|
| tall_box |24.13% /+16.31%|**86.94% /−72.17%**|**86.84% /−77.57%**|

Tutti i valori del motore sono finiti e non negativi. I controlli negativi
con immagini zero e2×reference falliscono sia NRMSE sia bias in tutte le ROI.
Le singole misure perframe e la variazione temporale sono nel JSON completo.

La regione leak, definita geometricamente come floor occluso dai box verso il
centro dell'emettitore, contiene solo1pixel dopo erosione: **non certificabile**
(minimo16). Non è stata ampliata dopo aver visto i risultati. Thin walls,
probe inside wall, moving lights e disocclusion richiedono altri snapshot.
La recovery è preregistrata entro128frame alle stesse soglie statiche e con
residuo del vecchio stato≤10%; questa scena statica non la verifica.

## Riproduzione ed evidenza

Strumenti: `tools/f12_oracle_{validate,texture_check,render,compare}.py`.
Usano esclusivamente un Python già dotato di Mitsuba3.9.1 e NumPy. Non installano
pacchetti, non avviano il renderer e non usano GPU. Applicare prima la patch
all'adapter o passare una sua copia corretta con `--reference-script`.

Dati numerici committati:
`docs/results/F12-cornell-oracle-M5Max-2026-10-07.json`.
Dati grezzi, sorgente adapter eseguito, PFM full/direct/indirect per ogni seed,
maschere, snapshot, negativo precedente e log:
`build/f12-oracle-v1/` nel worktree `f9-proxy`.
`reproduce.sh` contiene i comandi parametrizzati.
Mosaico solo illustrativo: reference / DDGI / cache / ReSTIR, stessa scala6×
e conversione sRGB, nel file `quality/comparison-reference-ddgi-cache-restir.png`.
Le metriche usano sempre i PFM lineari originali.

Fonti primarie: [Mitsuba3.9.1 path-depth](https://mitsuba.readthedocs.io/en/v3.9.1/src/generated/plugins_integrators.html),
[area emitter3.9.1 e scelta della proposta](https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/src/emitters/area.cpp).
