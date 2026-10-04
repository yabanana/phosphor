# Revisione generale dopo F8 — si può passare a OPT-2?

2026-10-04, `main` `769de09` (F8.4 in-process integrata), M5 Max 128 GB,
macOS 27.2 beta 2 (`26B5091g`), MetalFX 40.9. Richiesta del proprietario:
capire se l'episodio F8.4 ha lasciato danni e se si può avviare OPT-2.

## Esito in breve

**Il motore è sano; OPT-2 non è ancora pronta e, sui dati, non è la priorità.**
Quasi tutte le batterie storiche passano su `main`. La revisione trova due
difetti reali nell'attrezzatura che OPT-2 userebbe come rete di sicurezza:
l'equivalenza delle varianti e i tempi per pass. Entrambi nascono in F7/F8 e
vanno corretti prima di qualunque OPT. Le misure mostrano inoltre che il tempo
GPU non sta dove OPT-2 interviene.

## Batterie eseguite su `main`

| Controllo | Esito |
|---|---|
| `ctest` Debug e Release | 100% (438 casi, 1.773.956 asserzioni) |
| `visual_check` (API + shader validation), senza flag e con `--debug-graph-transients`, `--debug-split-encoding`, `--debug-async-compute` | 0 pixel, 0 messaggi |
| `f6_check.sh` completa (7 modalità mesh, controlli negativi, overflow, eventi, switch/resize) | PASS |
| `archive_check.sh` (riferimenti Release rigenerati da `main`) | PASS |
| `variant_check.sh` percorso mesh | 284 compatibili (peggiore 35 pixel), 52 negativi: PASS |
| `variant_check.sh` percorso indexed | **28 FAIL**: bench 4, modalità di debug 1 e 2 (vedi sotto) |
| `hitch_check.sh`, `hot_reload_check.sh` | PASS |
| `graph_scenarios.sh` (pixel e validation, 1 round) | 0 pixel diversi, 0 messaggi |
| `leaks --atExit` nativo con switch | 0 leak |
| `f7_f8_check.py` funzionale / qualità / contesto | 26/26, 16/16, 24/24 |
| `metalfx_lifetime_check.py`, soak 80 ricreazioni | PASS (PR #16) |
| CI `main` (Linux core + syntax, macOS build) | verde |

Il codice non contiene TODO/FIXME; i rinvii residui sono pianificati (F16
trasparenze, F22 streaming). Nessun link rotto o JSON non valido nei documenti.

## Difetti trovati

1. **Varianti forward non equivalenti su Sponza (aperto, causa individuata).**
   Dal commit F7 `70a64d8` la pipeline generica e quelle specializzate di
   bench 4 (Sponza, caricata da quando F7 ha aggiunto l'asset) differiscono
   nelle modalità di debug 1/2: 48.081 e 69.903 pixel, delta fino a 49/52.
   La modalità illuminata resta nel contratto (9–11 pixel, delta 1); il
   percorso mesh passa. Prima di F7 (`d9cfacf`, stessa scena) la differenza
   è 0. Il "peggiore un pixel" del handoff F7/F8 non si riproduce con Sponza.
   Esperimenti (tutti temporanei, shader ripristinati e verificati a 0 pixel):
   stesso triangolo vincente per pixel (`primitive_id`), UV dei pixel visibili
   identiche al bit, layout del quad identico; coincidono campionando a mip 0;
   divergono con LOD hardware o gradienti espliciti, anisotropia 8 o 1, mip
   lineare, nearest o nessuno, archivio AOT o no, warmup 30 o 400. Le derivate
   (`dfdx`) differiscono nei bit bassi in ~2% dei pixel anche nei quad pieni;
   derivate esplicite via `quad_shuffle_xor` divergono ugualmente e leggono
   valori errati dai lane inattivi. Causa: i valori dei lane non visibili
   del quad (helper/occlusi), con cui GPU e driver calcolano le derivate, non
   coincidono fra le due compilazioni della stessa funzione; non è
   controllabile dallo shader. Correzione vera: derivate analitiche anche
   nel forward (come il resolve compute), prerequisito registrato per
   OPT-2.2/OPT-2.11. Il percorso HDR di prodotto (visibility + resolve con
   derivate analitiche) non è interessato.
2. **Tempi per pass doppi nella catena post F8 (corretto, `8cdb4a8`).** Nei frame F8 la
   somma delle unità supera lo span: 21,83 contro 11,35 ms (Many Lights), 0,90
   contro 0,63 ms (Sponza nativo). `Luminance histogram` riporta sempre
   resolve + ~0,03 ms (10,489/10,459; 0,291/0,252; 0,211/0,182): il suo
   intervallo parte prima del `Material resolve`, che viene contato due volte.
   Frame time e span sono corretti; nel frame forward somma e span coincidono
   (0,302/0,303). Causa: il clear dell'istogramma gira accanto al resolve e
   finisce prima; l'istogramma partiva dalla sua fine. Ora un'unità parte
   dalla fine più tarda della sua catena: somma = span (0,628/0,628;
   2,137/2,137; 11,378/11,378), istogramma 0,015–0,025 ms; controllo a costo
   noto lineare (2000/8000 iterazioni: 1,38/5,53 ms), test di regressione.
3. **Riferimenti visivi superati (corretto).** I riferimenti F6 precedevano
   la correzione F7 di normali e tangenti specchiate: attribuiti per bisezione,
   rigenerati, batteria F6 40/40 ([dettagli](2026-10-04-metalfx-cycle-root-cause.md)).
   Anche `build/reference` (default di `visual_check`) era dell'era F3: ora è
   rigenerato da `main` e coincide con `reference-f6base` (0 pixel); i
   precedenti restano in `build/reference-f3-era`.
4. **Registro hardware senza F7/F8 (corretto).** Aggiunta la riga in
   [HARDWARE_VALIDATION](../plans/HARDWARE_VALIDATION.md).
5. **Uscita F8 "misure per ricalibrare i budget": parziale.** Le misure
   esistono (sotto e nel handoff F7/F8); la ricalibrazione dei budget non è
   registrata. `docs/perf-history.csv` è fermo al 2026-09-29 (F4).

Non bloccanti: 9,8 GB di artefatti in `build/` (2,4 GB `graph-select`).

## Baseline prestazionale e dove va il tempo GPU

`tools/f7_f8_bench.py`, Release, 1920×1080, 120 warmup + 600 frame, tre
repliche ruotate, offscreen, mediane. Rispetto alla baseline PR #14 tutti i
casi sono fra −0,5% e −1,9%: nessuna regressione.

| Caso | Frame ms | p95 | Unità GPU dominante |
|---|---:|---:|---|
| Sponza forward | 0,303 | 0,352 | Forward 93% |
| Sponza visibility | 0,593 | 0,692 | Material resolve |
| Sponza HDR nativo | 0,631 | 0,733 | Material resolve 0,25 ms |
| Sponza HDR + MetalFX | 2,140 | 2,240 | **Temporal reconstruction (MetalFX) 1,60 ms, 69%** |
| Cornell + MetalFX | 2,066 | 2,187 | **MetalFX 1,77 ms, 83%** |
| Many Lights 1.024 + MetalFX | 11,354 | 11,580 | **Material resolve ~10,4 ms** |

- Nei frame temporali il costo è MetalFX, codice Apple chiuso: OPT-2 non lo
  tocca. Gli shader nostri pesano ~0,5 ms; il −15% del tempo GPU totale
  richiederebbe di dimezzarli tutti.
- Many Lights è il ciclo su tutte le luci senza culling: un limite
  algoritmico che la roadmap assegna a F11 (cluster di luci 3D, ReSTIR;
  uscita "Many Lights stabile"), non a registri o occupancy.

## Raccomandazione

1. Correggere i due difetti aperti (attribuzione dei tempi per pass della
   catena post; equivalenza delle varianti nei modi di debug), con controlli
   negativi. Stima: circa una giornata, con incertezza sul secondo.
2. Registrare la baseline OPT-4.16 con i tempi per pass corretti
   (`perf_record` incluso), come prevede [SEQUENCING](../plans/SEQUENCING.md).
3. Sui dati attuali **non avviare OPT-2**: il collo di bottiglia prioritario
   è l'illuminazione (proseguire verso F9–F13/F11) o, al più, un esperimento
   mirato sul ciclo luci del resolve. Rivalutare OPT-2 dopo F13, quando il
   frame con luce reale renderà caldi shader nostri.

Dati locali: `build/review/` (batteria, bench, repro varianti, mappe delle
differenze).
