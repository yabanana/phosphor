# F13/F14 — brief di chiusura (revisione dell'aggregatore, 2026-11-10)

Stato: **piano operativo, non accettazione di fase.** Nessuna casella della
roadmap cambia con questo documento. Le decisioni D1–D3 sono state prese dal
proprietario il 2026-11-10 e sono registrate in §4. Sorgente esaminata: branch
`codex/f13-f14-dynamic-closure` (03d2874), binario `build/closure-release/phosphor`
di quel worktree (sha256 `0e2c2260ca4d052a…`), M5 Max, macOS 27.3.
Evidenze grezze di questa revisione: `build/f13-f14-closure-review-20261110/`
nel checkout principale (cartella ignorata da git).

## 1. Diagnosi in breve

Il lavoro non era fermo per mancanza di un riferimento su Sponza. Era fermo
perché tre dei cinque gate aperti sono formulati in modo che nessun risultato
possa chiuderli, e perché il corpus congelato di F13.6 non risulta mai eseguito:
non ne ho trovato traccia in nessuna cartella `build/` né nei documenti.
Eseguito oggi per la prima volta, in 98 secondi ha mostrato un difetto reale
del motore (§2.1) che nessun riferimento su Sponza avrebbe isolato.

| Gate | Causa reale del blocco | Si chiude con |
|---|---|---|
| F13.6 | Banco sbagliato (Sponza texturizzata) e corpus procedurale mai eseguito; il primo caso rivela un doppio conteggio di energia | Correzione del motore, poi clip di riferimento sul corpus procedurale (§3.1) |
| F13.3 | La soglia «1% per canale» senza pavimento assoluto è irraggiungibile per costruzione (§2.2) | Nuovo criterio D1 (deciso), fixture aggiornata e ammissione del nativo |
| F14 | Tutti i controlli passano; resta «pending» per una regola del runner che il piano non prevede, più alcuni casi del piano mai scritti (§3.2) | Correzione del runner e casi mancanti |
| F9 58/60 | Il raggio passa a 0,056 ULP da un bordo: l'esito non è determinato in float32 (§2.3) | Criterio di errore all'indietro nell'oracolo (D3, deciso) |
| 16,67 ms | 2,05 ms di pass di verifica sempre attivi e 9,25 ms di denoise (§2.4) | Verifiche fuori dal frame di prodotto, poi circa 4 ms dal denoise (D2: il gate resta a 1080p nativo) |

## 2. Fatti misurati in questa revisione

### 2.1 F13.6 — il corpus congelato, eseguito

Piano congelato e poi eseguito con il runner invariato:

```sh
python3 tools/f13_f14_check.py --binary build/closure-release/phosphor --out <dir> \
  --only mirror-custom --only mirror-rt8-control
python3 tools/f13_f14_check.py --run --manifest <dir>/frozen-manifest.json \
  --binary build/closure-release/phosphor --out <dir>
```

Due casi da 300 frame a 640×360 con validazione API e shader: controlli
funzionali PASS, 98 s in tutto. Il confronto `mirror-custom` contro
`mirror-rt8-control` invece **fallisce** di due ordini di grandezza: RMSE p95
8,35 e massimo 9,91 contro 0,06 e 0,15; flicker 0,110 contro 0,02.

La causa è un **doppio conteggio dell'energia speculare degli emettitori**.
Nell'immagine del pannello emissivo dentro lo specchio il valore analitico è
F0·Le = 0,95·(8; 0,4; 0,2) = (7,6; 0,38; 0,19), più un piccolo termine diffuso.

| Configurazione (media su 140–200 frame, regione 19×14 px) | R medio | Mediana di R per frame |
|---|---:|---:|
| Solo luce diretta `brute`, riflessi spenti | 7,34 | 0,29 |
| `brute` + riflessi RT ×8, grezzo | 15,19 | 8,18 |
| ReSTIR DI + riflessi RT ×8, grezzo | 15,73 | 8,48 |
| `mirror-rt8-control` del runner | 15,58 | 8,19 |
| `mirror-custom` del runner (denoise custom) | 11,75 | 11,80 |

La luce diretta valuta `diBRDF` completa (diffuso più GGX) per ogni emettitore
campionato, anche su un ricevitore quasi speculare (`shaders/light_cluster.metal`,
`shaders/restir_common.h:122`). Il pass dei riflessi traccia poi un raggio sul
lobo GGX e somma l'emissione di ciò che colpisce (`shaders/reflections.metal`,
`reflectionSecondary`: `result=max(emission,0)`). Le due stime dello stesso
trasporto vengono sommate senza pesi MIS: 2,05× l'energia nel grezzo. Il
denoise custom ne taglia una parte con il clamp dei firefly e si ferma a 1,55×.
Vale anche con ReSTIR, quindi riguarda il percorso di prodotto. Il piano F13
vieta esplicitamente di raddoppiare l'energia fra riflessi e fallback.

Non verificato: se lo stesso doppio conteggio esiste per il sole quando i
raggi di riflesso mancano la geometria e leggono un cielo che contiene il disco
solare. Va controllato con lo stesso metodo.

### 2.2 F13.3 — errore dell'SDK MetalFX per canale

Programma: `build/f13-f14-closure-review-20261110/metalfx-channels/sdk_hdr_channels.mm`,
derivato da `bench/metalfx_denoised/sdk_hdr.mm` (stessa API pubblica, stesse
guide, stessi 96 frame). Riporta l'errore per canale invece del solo massimo.
I due casi congelati sono riprodotti: 0,39% e 5,1758%.

«Esposto» = valore impacchettato / preExposure × esposizione, cioè ciò che
l'SDK vede rispetto al bianco di esposizione.

| Caso | Valore esposto | Errore relativo massimo |
|---|---|---|
| Congelato «wide», rosso | 32 | 1,81% |
| Congelato «wide», verde | 0,011 | 1,46% |
| Congelato «wide», blu | 0,0056 | **5,18%** (il numero del gate) |
| Grigio | 32 | 4,65% (media −1,5%) |
| Grigio | 2,8 | 0,83% |
| Grigio | 0,36 – 0,5 | 0,20 – 0,34% |
| Grigio | 0,016 – 0,125 | 0,29 – 0,49% |
| Grigio | 0,0020 | 8,8% |
| Grigio | 0,00024 | 86,8% |
| Rosso a 0,5 con verde/blu a 0,0078 / 0,0020 / 0,00049 | — | 1,46% / 5,08% / 21% |

Lettura: l'SDK ha un **pavimento di errore assoluto di circa 1–3·10⁻⁴ del
bianco di esposizione** (circa un codice sRGB a 8 bit vicino al nero). Sui
grigi resta entro lo 0,5% fra 0,01× e 0,5× il bianco. La fixture congelata
mette il rosso a 32× il bianco e il blu a 0,0056×: fallisce una soglia
relativa pura in entrambe le direzioni, per costruzione.

Scansione delle alte luci (`sdk_hdr_scan.mm`, `scan.jsonl` nella stessa
cartella), grigi con valore esposto crescente. Due parametrizzazioni diverse
(valore impacchettato variabile oppure esposizione variabile) danno gli stessi
numeri: l'errore dipende solo dal valore esposto.

| Valore esposto | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Errore massimo % | 0,00 | 0,29 | 0,54 | 0,49 | 0,98 | 2,15 | 2,34 | 2,25 | 4,95 | 4,69 | 10,16 |
| Bias medio % | 0,00 | −0,05 | −0,26 | −0,08 | −0,20 | −1,17 | −0,35 | +0,36 | −1,40 | −1,56 | −4,62 |

Con colore (R; R/4; R/16): a 4× il canale peggiore è allo 0,78%; a 32× il
rosso è al 7,23%. Il limite dell'1% tiene fino a 4× il bianco con margine
circa doppio, è al filo a 6× e cade a 8×. Riportando tutti i 44 casi misurati
sulla curva SDR del motore (`acesFit` più codifica sRGB, stima per canale
fatta fuori dal motore), l'errore peggiore vale 0,54 codici a 8 bit. Il
percorso EDR con headroom comprime meno le alte luci e non è stimato. Scegliere un'altra esposizione sposta il fallimento sugli altri
canali, non lo elimina. Non è una prova di difetto dell'SDK e non è
un'ammissione: è il profilo misurato su questa versione, da rimisurare a ogni
aggiornamento di sistema.

### 2.3 F9 — testimone bench8 frame48/ray64

`build/f13-f14-closure-review-20261110/f9-edge/edge_witness_exact.py` ricalcola
il testimone in aritmetica razionale esatta sugli stessi ingressi float32.

- Il punto d'impatto dista 2,13·10⁻⁷ m dal bordo; un ULP della coordinata z
  dell'origine vale 3,8·10⁻⁶ m. Il raggio passa a **0,056 ULP** dal bordo.
- Spostando l'origine di ±1 ULP per componente, 12 casi su 27 diventano
  mancati esatti; 9 su 27 per la direzione; 12 su 27 per la traslazione
  dell'istanza.

L'esito «colpito» non è determinato alla precisione degli ingressi. Un
intersettore float32 che riporta «mancato» dà la risposta esatta di un
problema perturbato di meno di un ULP. Il test passava su macOS 27.2 per caso.

### 2.4 Budget — dove vanno i 21,6 ms

Dal report `build/f13-f14-refinement-perf/1-B/renderer.json` del worktree di
sviluppo (Sponza 1080p completo, GPU p50 21,61 ms, p95 22,69 ms, somma dei
pass 20,63 ms):

| Gruppo | ms (p50) | Quota |
|---|---:|---:|
| Denoise custom, temporale + atrous, 12 pass | 9,25 | 45% |
| Riflessi RT | 3,42 | 17% |
| **Pass di verifica dentro il frame misurato** | **2,05** | **10%** |
| RTAO | 1,20 | 6% |
| Luce diretta | 1,10 | 5% |
| Volumi (cielo, nebbia, nuvole) | 0,83 | 4% |
| Ombre solari | 0,64 | 3% |
| GI (DDGI denso) | 0,56 | 3% |

I pass di verifica sono `F13 denoise {1,2,3} independent state check` (1,27 ms),
`F13 independent receiver motion input check` (0,39 ms) e
`F13 independent raw specular reduction` (0,39 ms, da confermare che sia solo
diagnostico). In `denoise_passes.cpp` `addChecks` viene chiamato senza
condizione; la luce diretta invece li crea solo con `debugLighting`. Il comando
di misura non passa `--debug-lighting`, quindi il budget di prodotto paga la
propria verifica. Toglierli non cambia un pixel e porta il p95 verso 20,6 ms.
Restano circa 4 ms, da cercare nei 9,25 ms di denoise.

## 3. Risposte alle due domande dello sviluppo

### 3.1 Qual è il caso più piccolo di F13.6 con una clip accettata dal runner

**`mirror-custom` insieme a `mirror-rt8-control`.** Scena e camera sono
statiche, i materiali sono soli fattori con texture 1×1, quindi la parità dei
materiali vale per costruzione (`reference.py:206`) e basta **un solo frame di
riferimento convergente**, ripetuto con gli stessi 300 ID.

Cosa serve perché il runner esca con 0:

1. Piano congelato con `--only mirror-custom --only mirror-rt8-control`
   (il `--only` vale solo alla creazione del piano). Entrambi i casi hanno
   bisogno di una voce in `--references`, altrimenti restano «pending».
2. File `refs.json`:
   `{"mirror-custom": {"linear": "<dir>", "provenance": {"validated": true,
   "independent": true, "manifest_sha256": "<sha256 dei byte di frozen-manifest.json>"}},
   "mirror-rt8-control": { …stessa clip… }}`.
3. Nella cartella: `frame-000000.pfm` … `frame-000299.pfm`, RGB Float32
   lineare, 640×360, tutti finiti.
4. Riferimento prodotto **da chi non ha scritto il renderer** (lo faccio io):
   esportazione `--export-reference` del frame, Mitsuba 3.9.1 `llvm_ad_rgb`
   (ambiente già presente in `build/sponza-oracle-venv` più `build/reference-llvm`),
   raggi primari al centro del pixel, due semi con il tetto del 2% invariato.
   Il modello `principled` non va usato: serve un BSDF che implementi il
   contratto del motore (Lambert·(1−F)·(1−metallic) più GGX con Smith correlato
   e Schlick), validato sui controlli chiusi già esistenti (1−ln 2, irradianza
   analitica).
5. Scala della ROI dichiarata **prima** dell'esecuzione dal bianco fisico della
   scena (pavimento al sole: 12·cosθ·0,65/π ≈ 2,1), con `--roi-config` alla
   creazione del piano. La scala 1,0 attuale è un segnaposto scritto prima di
   qualunque misura.

Due cose da decidere prima di congelare, perché oggi renderebbero il confronto
privo di senso:

- **Il difetto di §2.1 va corretto prima.** Finché il controllo a 8 raggi ha
  firefly da 1500 e il custom ha il 55% di energia in più, nessuna clip di
  riferimento può passare.
- **Caustica nella ROI.** Il sole illumina la faccia dello specchio e il suo
  riflesso cade sul pavimento davanti, dentro la ROI. Un path tracer completo
  aggiunge lì circa +2,0 di radianza diffusa che il motore non modella per
  progetto; il motore ne campiona solo la parte speculare, con varianza
  enorme. Proposta: ROI separate («superficie dello specchio» e «pavimento»),
  spazio dei cammini del riferimento dichiarato uguale a quello del motore, e
  regolarizzazione della roughness sulla luce diretta ai colpi secondari.

Ordine dopo `mirror`: `disocclusion` (superficie che torna visibile, camera in
moto e taglio al frame 239: 241 frame distinti), `moving-light` (gradino
dell'emissivo al 119), `roughness`. Sono le quattro voci della riga F13.6 della
roadmap. Sponza non serve a questo gate e va tolta dal percorso di chiusura.

### 3.2 Mappa delle prove F14 e casi scoperti

Ultima esecuzione completa: `build/f13-f14-f9-refinement/checks/volumes-stable`,
nove casi da 300 frame, tutti PASS su controlli funzionali e numerici.

| Task | Verifica chiesta dal piano | Prove esistenti | Scoperto |
|---|---|---|---|
| F14.1 | Sole allo zenit e all'orizzonte, camera alta, parametri estremi; riferimento integrale, nessun NaN | `atmo-zenith`, `atmo-horizon`, `atmo-space`: 2400 confronti ciascuno su trasmittanza, multi-scattering, sky-view e disco solare; negativi `lut` e `units` | **Parametri estremi** (manca anche l'opzione per impostarli) |
| F14.2 | Densità costante analitica, slicing, luce mobile, disocclusioni; costo per volume | `fog-homogeneous-oracle` (Beer-Lambert esatto, 3300 confronti); `all-volumes-moving` solo funzionale; negativo `light` | **Slicing**, **qualità con luce mobile**, **disocclusione con nebbia**, **costo del volume** nel perf-log |
| F14.3 | Volo dentro la nube, controluce, bordo schermo; ghosting e conservazione della copertura | `clouds-reconstructed` contro `clouds-full` a 2000 m, dentro lo strato 1500–4000 m: p95 0,00108, flicker 0,00015, ghost 0,064, recupero 0 frame, bias 2·10⁻⁵ | **Volo vero** (la camera si sposta di 1,8 m), **controluce**, **bordo schermo** con rotazione |
| F14.4 | Transizione giorno/notte ripetibile, salto temporale con reset | `atmo-time-jump` (orologio solare esatto ed epoche), positivo e negativo della storia | **Ripetibilità** fra due esecuzioni; **notte, luna e stelle** verificate solo da test CPU |

Il motivo per cui il runner non dichiara mai completa la selezione F14 è la
riga 408: ogni caso con cattura e senza clip esterna è «pending». Il piano F14
indica come riferimenti l'integrazione numerica su pochi raggi per le LUT e il
**full-rate per la ricostruzione temporale**. Una clip offline indipendente di
atmosfera e nuvole non è richiesta da nessuna parte. Correzione: ogni caso
dichiara il tipo di riferimento previsto dal piano (oracolo numerico, coppia
full-rate, clip indipendente) e resta «pending» solo se manca quello. Vale
anche per i casi F13 `*-raw`, `*-rt8-control` e `mirror-ssr`, che sono
controlli o diagnostici e non devono a loro volta superare un riferimento.

## 4. Decisioni del proprietario (prese il 2026-11-10)

- **D1 — criterio radiometrico di F13.3: pavimento assoluto.** La soglia
  «1% per canale» è sostituita da:
  - valore esposto fino a **4× il bianco**: errore ≤ max(1% del valore;
    5·10⁻⁴ del bianco di esposizione). Margini misurati: 0,83% nel caso
    peggiore fino a 4×; pavimento 2,9·10⁻⁴.
  - sopra 4× il bianco: errore ≤ 2/255 dopo il tone map SDR del motore e la
    codifica sRGB. Stima attuale del caso peggiore: 0,54/255. Va rimisurato
    dentro il motore, e il percorso EDR va misurato a parte.
  - i limiti in lineare delle alte luci restano dichiarati: fino al 2,3% a
    8–16×, 5% a 24–32×, 10% a 64×, con bias medio fino a −4,6%.
  - tutto è legato alla versione di MetalFX misurata e va rimisurato a ogni
    aggiornamento di sistema.
- **D2 — il budget di 16,67 ms resta a 1080p nativo**, con tutte le funzioni
  e gli stessi raggi. La strada è: verifiche fuori dal frame di prodotto
  (−2,05 ms), poi circa 4 ms dai 9,25 ms di denoise. Non è garantito che si
  trovino senza toccare raggi o risoluzione; se dopo le misure il margine non
  c'è, il gate resta fallito e torna al proprietario.
- **D3 — oracolo F9: due esiti ammessi.** Per i soli raggi entro il limite di
  arrotondamento float32 da un bordo vale «colpito» oppure il risultato esatto
  senza quel triangolo; ogni altro esito resta FAIL. I raggi ambigui vengono
  contati e limitati, con controlli negativi (un raggio spostato dentro il
  triangolo deve colpire, uno spostato fuori deve mancare).

- **D4 — ghost F13 sull'errore che cambia** (2026-11-11), vedi §14.
- **D5 — regolarizzazione dichiarata k = 8** (2026-11-11). Nella connessione
  alla luce dopo un lobo riflesso campionato, α' = max(α_secondario,
  8·α_camera), nel motore e negli strumenti di riferimento (aec6ddc). È il
  k più piccolo con cui `roughness` passa su tutti i semi; le altre fixture
  sono identiche a k = 1. Dati in §16.

## 5. Ordine di lavoro proposto

Sviluppo (worktree `codex/f13-f14-dynamic-closure`):

1. Correggere il doppio conteggio emissivo (MIS fra campionamento delle luci e
   del BSDF, oppure proprietà del lobo speculare per roughness). Criterio:
   immagine del pannello = F0·Le più diffuso entro il 3%, in grezzo e in custom.
   Controllare lo stesso per il sole.
2. Mettere i pass di verifica del denoise dietro `--debug-lighting`, poi
   cercare i circa 4 ms restanti nel denoise (D2), con misure A/B/A.
3. Runner: tipo di riferimento per caso (§3.2); ROI di `mirror` separate.
4. F14: casi mancanti della tabella §3.2.
5. F13.3 secondo D1: fixture con il nuovo criterio, contratto di esposizione
   in ingresso all'SDK e ammissione del nativo legata alla versione.

Aggregatore (questa chat):

1. Oracolo F9 secondo D3 e matrice a 60 casi.
2. Adattatore BSDF e clip di riferimento indipendenti, da `mirror` a `roughness`.
3. Tutte le esecuzioni di verifica, le misure A/B/A e l'integrazione.

## 6. Da non ripetere

- Riferimenti su Sponza per chiudere F13.6: la parità dei filtri di texture
  non è dimostrabile e il gate non la richiede.
- Altre varianti di esposizione, formato o maschera per far passare la fixture
  «wide» attuale: §2.2 mostra che il risultato non dipende da quelle scelte.
- Leggere l'esito aggregato del runner come stato della fase: `phase_accepted`
  e `corpus_complete` sono costanti `False` nel codice.
- Micro-ottimizzazioni aritmetiche dei kernel prima di aver tolto dal frame i
  2 ms di verifiche.

## 7. Clip di riferimento indipendenti (aggiornamento 2026-11-10, sera)

Strumenti sul branch `review/f13-reference-clips` (worktree
`.claude/worktrees/f13-reference`, commit 715bd1d):
`tools/f13_reference.py`, `tools/f13_reference_clip.py`,
`tools/battery_pause.py`, `tests/test_f13_reference_clip.py` e
`tests/test_battery_pause.py` (registrati in CTest). Evidenze in
`build/f13-reference-clips-20261110/` del checkout principale.

**Modello del riferimento.** Mitsuba 3.9.1 `llvm_ad_rgb` per raggi e
campionamento delle luci; BRDF e integratore scritti dal contratto del motore
(Lambert·(1−F)·(1−metallic) + GGX con Smith correlato e Schlick, roughness
≥ 0,04, α ≥ 0,002). Materiale completo al vertice della camera e al primo
colpo dopo il lobo speculare, riduzione diffusa F12 oltre. Ogni emettitore è
contato una volta (pesi MIS). Esclusi per costruzione, come nel motore: caustiche
attraverso uno specchio e inter-riflessioni speculari oltre la prima. Ambiente:
il cielo dello snapshot. Raggi al centro del pixel, due semi, convergenza
≤ 2% su frame e su ogni regione dichiarata.

**Validazione.**
- 27 controlli in forma chiusa: conduttore bianco ruvido 1 − ln 2, energia
  campionata contro quadratura per 4 roughness × 3 angoli, densità VNDF,
  termine di Lambert.
- Posizioni primarie contro l'esportazione del motore su 23 snapshot: entro
  1,1·10⁻⁵ m (7,3·10⁻⁵ m sulle sfere). Da 0 a 3 pixel per frame di silhouette,
  ammessi solo se un raggio spostato di 0,01 pixel riproduce l'esito del motore.
- Script di `disocclusion` e `moving-light` ricostruiti contro 20 snapshot del
  motore entro 4·10⁻⁷. Taglio al frame 239 e gradino al 119, come nel runner.

**ROI congelate** (`roi-config.json`) prima di leggere immagini del motore:
scala 2,0 = radianza analitica del pavimento al sole (12·0,848·0,96·0,65/π ≈
2,02, arrotondata per difetto); per `mirror` tre regioni (`primary-signal`,
`mirror-surface`, `floor-front`); soglie F8 invariate.

**Primo confronto reale** (`compare-mirror-baseline`, runner invariato,
binario attuale con il difetto di §2.1). Nessun caso è più «pending».

| Caso | `floor-front` p95 | `mirror-surface` p95 | Esito |
|---|---:|---:|---|
| `mirror-custom` | 0,033 | 0,214 (bias +0,06) | FAIL solo sullo specchio |
| `mirror-rt8-control` | 0,058 | 6,43 | FAIL (firefly) |
| `mirror-raw` | 0,150 | 6,43 | FAIL (rumore e firefly) |

Scomposizione di `mirror-custom` sullo specchio: il 96% dell'errore quadratico
viene dall'immagine del pannello emissivo e dai suoi bordi (11,9 contro 7,8 del
riferimento; bordi allargati di 1–2 pixel dal denoise), il 4% dall'ambiente non
dichiarato (difetto 2 sotto), che da solo vale 0,042 contro la soglia 0,06.

**Difetti del motore che il riferimento rende misurabili.**
1. Doppio conteggio emissivo (§2.1). È anche l'origine dei firefly dei casi
   grezzi: la luce diretta valuta GGX con α = 0,0016, dove D arriva a circa
   1,2·10⁵. Correzione richiesta: un singolo conteggio pesato fra
   campionamento della luce e del lobo speculare.
2. Tre ambienti nello stesso frame: la GI usa il cielo esportato (nero), i
   raggi di riflesso che mancano usano una costante (0,03; 0,035; 0,045), lo
   sfondo visibile è (0,02; 0,025; 0,035), e con i riflessi spenti il resolve
   aggiunge il gradiente cielo/suolo legacy (0,30/0,36/0,45). Correzione
   richiesta: un solo ambiente dichiarato ed esportato nello snapshot.

3. Sole puntiforme nel resolve primario: `evaluateLight` valuta GGX verso una
   sola direzione, mentre ombre e raggi di riflesso usano il disco dichiarato
   (raggio angolare 0,00465 rad). Con α = 0,002 il lobo è largo quanto il
   disco. Confronto `compare-roughness-baseline`: `roughness-custom` ha p95
   0,253 e bias +0,008; il 96% dell'errore quadratico sta in 24 pixel di
   riflesso del sole (picco 659 contro 788, un pixel di bordo 0,6 contro 117,
   gli altri entro l'1%). Senza quei pixel l'RMSE normalizzato vale 0,051.
   Correzione richiesta: integrare il disco nel termine speculare del sole
   (per esempio allargando α del raggio angolare con la normalizzazione
   d'energia corrispondente), verificata su questa clip.

**Clip.** `mirror`: 1 stato, convergente a 16 384 campioni per seme (NRMSE
fra semi 1,0% ROI, 0,39% specchio, 1,1% pavimento). `roughness`: 1 stato, convergente a 8192 campioni
per seme (1,6%). `disocclusion`: 240 stati (frame 239–299 condividono lo stesso), ogni
stato converge a 6144 campioni, circa 12 ore. `moving-light`: 300 stati, in
coda. Le clip dinamiche coprono soltanto la finestra delle ROI più 2 pixel; il
report lo dichiara e fuori dalla finestra i pixel sono zero.

Uso, dopo la correzione del motore e un nuovo piano congelato:

```sh
python3 tools/f13_f14_check.py --binary <bin> --out <plan> --only 'mirror-*' --roi-config build/f13-reference-clips-20261110/roi-config.json
python3 tools/f13_reference_clip.py refs <plan>/frozen-manifest.json --case mirror-custom=build/f13-reference-clips-20261110/mirror/clip ... --out <plan>/refs.json
python3 tools/f13_f14_check.py --run --manifest <plan>/frozen-manifest.json --binary <bin> --out <plan> --references <plan>/refs.json
```

Le clip restano valide finché fixture, esportazione e script non cambiano: lo
sviluppo non deve modificare `src/testbench/reflection_validation.*` senza
rigenerarle.

## 8. F9 con il criterio D3 (2026-11-10, notte)

Branch `review/f9-edge-oracle` (worktree `.claude/worktrees/f9-oracle`,
commit 699f5af e 229e0fd con il documento
`docs/implementation/F9-D3-EDGE-ORACLE-2026-11-10.md`), sopra
`codex/f13-f14-dynamic-closure`.

- `RtReference::check`: se la GPU non concorda con il colpo del riferimento
  (mancato, oppure un colpo più lontano su un altro triangolo) e quel triangolo
  dista dal proprio bordo meno del budget d'arrotondamento FP32 degli ingressi
  (γ₈·‖|vertici trasformati| + |o| + |t·d|‖, limitato dalla tolleranza S0 in
  metri), la GPU deve coincidere in modo stretto con il risultato esatto senza
  quel solo triangolo. Nessun'altra tolleranza cambia.
- `RtChecker`: i raggi ammessi così sono contati (`edgeExclusions`) e limitati
  a max(1, controllati/1000) per frame; oltre il tetto il frame fallisce. Il
  conteggio compare nella riga per frame e nel totale `RT checks … | edge-excluded rays N of M`
  (formato ancora compatibile con `tools/f9_check.py`).
- Test doctest sul testimone reale: mancato GPU ammesso; raggio spostato
  all'interno (1,1·10⁻³ m) rifiutato; con un secondo bersaglio dietro, il
  mancato è rifiutato e il colpo sul secondo ammesso; tetto per frame.
  Controllo negativo: con la fascia resa infinita i test dell'interno
  falliscono; codice ripristinato. CTest 9/9.
- Matrice F9 completa (stesso driver di `build/f13-f14-f9-refinement/matrix.py`,
  binario `build/release` del worktree, stessa toolchain Metal 32023.921):
  **60/60**. Raggi ammessi dal criterio in tutta la matrice: 1 in
  `native-bench8` e 1 in `apple9-bench8`, entrambi al frame 48, nessun altro.
  Evidenze in `build/f9-d3/` del worktree. Il primo giro ha dato 38/60 perché
  il worktree non aveva il link `assets/sponza`; dopo il link sono stati
  rieseguiti solo quei 22 casi.

## 9. Integrazione e ritmo di lavoro (2026-11-10, notte)

**Branch.** Entrambi i branch di revisione partono da 03d2874, la punta
locale di `codex/f13-f14-dynamic-closure`, che è 12 commit avanti rispetto
a `origin`. Non li ho pubblicati: il push porterebbe con sé quei 12 commit
dello sviluppo, non ancora verificati e mai pushati. I due branch toccano file
disgiunti. Lo sviluppo li integra con:

```sh
git merge review/f9-edge-oracle review/f13-reference-clips
```

Dopo l'integrazione, il runner F9 dello sviluppo deve dare 60/60 con lo stesso
conteggio di raggi ammessi (2 su 2 808 814). Chi modifica
`src/renderer/rt_reference.*` o `rt_check.*` deve rieseguire la matrice.

**Ritmo (richiesta del proprietario).** Il portatile è collegato a un
alimentatore da 80 W, che eroga circa 76 W. Misure da `ioreg`, senza root:
- a riposo il sistema assorbe 12 W e la batteria si ricarica a +64 W;
- ogni thread Mitsuba pesa da 4 a 7 W;
- con 18 thread il sistema assorbiva 86 W e la batteria perdeva 9 W, anche se
  macOS la segnava «in carica».

Regole:
- un solo lavoro pesante alla volta, al massimo 8 thread di rendering;
- una pausa dopo ogni stato, pari a un quarto del tempo di calcolo;
- ogni 45 minuti di calcolo una sosta finché la batteria non recupera 20
  punti (al massimo fino al 90%; la sosta finisce prima se il caricatore si
  ferma);
- il calcolo si blocca se il portatile viene staccato dalla corrente o se la
  carica scende sotto il 50%;
- fra un blocco di test e l'altro, `python3 tools/battery_pause.py --rest 300
  --gain 15`.

Con questo ritmo il completamento delle due clip dinamiche richiede circa
12–14 ore invece di 8.


## 10. Confronto di `moving-light` (2026-11-11)

Clip: 300 stati, tutti convergenti (4096–8192 campioni per seme). Piano
congelato `build/f13-reference-clips-20261110/compare-moving-light-baseline`,
runner e soglie invariati, binario `closure-release` di 03d2874 (ancora con i
difetti 1–3 di §7). Un solo ROI: `primary-signal`, scala 2,0, gradino emissivo
al frame 119.

| Caso | p95 | max | flicker | ghost (max) | recupero | Esito |
|---|---:|---:|---:|---:|---:|---|
| `moving-light-custom` | 0,111 | 0,137 ✓ | 0,0064 ✓ | 1,077 | 0 frame ✓ | FAIL su p95 e ghost |
| `moving-light-rt8-control` | 3,86 | 4,85 | 0,036 | — | 0 | FAIL (firefly) |
| `moving-light-raw` | 3,86 | 4,85 | 0,042 | — | 0 | FAIL (firefly) |

**Correzione di lettura.** Il pannello emissivo (slot 64) sta dietro la
camera, a z = 8,4. Quello che si vede è la sua immagine nello specchio di
fondo, cioè il caso di §2.1. Nei frame 0–118 il pannello è rosso intenso e nei
frame 239–299 è blu; nei frame 119–238 è quasi spento. L'errore segue il
pannello, non il sole.

**Scomposizione di `moving-light-custom`.** Le classi sono ricavate solo dal
riferimento; le posizioni vengono dall'esportazione del motore.

| Fase | RMSE/scala | Immagine del pannello | Bordi nello specchio | Ambiente | Pavimento e cubo |
|---|---:|---:|---:|---:|---:|
| pannello acceso (0–118, 239–299) | 0,093–0,098 | 67–71% | 18–20% | 8–9% | 3–4% |
| pannello spento (119–238) | 0,036 | 0% | 30% | 64% | 6% |

In percentuale sull'errore quadratico.

- **Immagine del pannello:** energia +17% rispetto al riferimento, bordi
  allargati dal denoise. Senza questa classe l'RMSE scende a 0,053; senza
  anche i bordi riflessi, a 0,033. Pavimento e cubo da soli valgono 0,018.
- **Ambiente:** il 78% dei pixel scuri è lo specchio che riflette il cielo
  nero dello snapshot. Lì il candidato vale (0,067; 0,080; 0,105), con il
  rapporto dei colori dell'emisfero legacy, contro circa 0,001 del
  riferimento. Il restante 22% è cielo diretto, dove il candidato vale il
  colore di sfondo (0,02; 0,025; 0,035) e il riferimento 0. È il difetto 2.
- **Bordi:** quasi tutto l'errore di bordo sta nell'immagine riflessa
  (orizzonte del pavimento, cubo e bordi del pannello visti nello specchio).
  La giunzione pavimento–specchio e le silhouette dirette sono già a livello
  del riferimento.

**Ghost.**
- *Frame 104, valore 1,077:* il 96% viene dall'immagine del pannello. Un pixel
  di bordo passa da 3,7 a 15,9 in un frame, mentre il riferimento resta
  intorno a 2. È un picco del doppio conteggio, non storia trascinata.
- *Frame 134–156, valori 0,42–0,50:* vengono dal bias d'ambiente nello
  specchio, proiettato su variazioni del riferimento molto piccole (energia di
  variazione 0,004–0,015). La metrica `support_ghost` amplifica un bias
  statico quando il riferimento cambia poco. Per chiudere non basta correggere
  il difetto 1: serve anche il difetto 2.

**Conseguenze per lo sviluppo.** Nessun difetto nuovo. `moving-light` conferma
i difetti 1 e 2 su una luce in movimento. Recupero dopo il gradino, flicker
ed errore massimo passano già. Dopo le correzioni, il criterio di verifica è
lo stesso piano con le stesse clip.

Analisi riproducibili (sola lettura) nella stessa cartella: `analysis.py`
(per fase e per frame), `classes.py` (per classe), `viz/` (immagini).

## 11. Chiusura in corso (2026-11-11, questa chat)

Il proprietario ha chiesto di chiudere davvero i punti aperti. Il lavoro è
sul branch `phase/f13-f14-closure` (worktree `.claude/worktrees/f13-f14-closure`,
base 03d2874 più i due branch di revisione). Ogni correzione è verificata
contro le clip indipendenti con runner e soglie invariati, poi committata.
Build con la toolchain di sistema 32023.922: la 32023.921 montata in
`/private/tmp` è sparita al riavvio.

| Commit | Correzione | Effetto misurato |
|---|---|---|
| f30a93b | speculare degli emettitori con geometria contato una volta (quota DI = smoothstep, quota RT = 1 − smoothstep) | specchio p95: raw 6,43 → 0,149, custom 0,214 → 0,087, rt8 6,43 → 0,065; energia dell'immagine del pannello da 2,05× a 1,03× |
| f3d728e | un solo ambiente dichiarato (raggi mancati, sonda analitica solo con contenuto reale, sfondo, termini ambientali di ripiego) | bias sullo specchio da +0,040 a +0,002; `mirror-rt8-control` passa su tutte le regioni; `moving-light-custom` da 0,111 a 0,072 |
| bb376ab | disco solare nello speculare primario (quadratura deterministica) | centro del riflesso solare 658 → 801 (riferimento 788); `roughness-custom` 0,253 → 0,180 |
| 5e94b3e | regolarizzazione della connessione alla luce dopo il lobo di camera, come dichiara il riferimento | `roughness-rt8` 1,02 → 0,51; medie a 4 frame dei riflessi interriflessi uguali al riferimento |
| ae44691 | runner: ogni caso giudicato col riferimento che il suo piano dichiara (clip indipendente, oracolo numerico, coppia full-rate, controllo, funzionale) | i casi raw/rt8/SSR e i controlli full-rate non bloccano più; i casi F14 con oracolo numerico non restano «pending» |

**Diagnosi aperte sul denoise custom (F13.4), misurate oggi.**
- *Sfocatura dei dettagli speculari nitidi.* L'à-trous speculare ha passi
  fissi (1, 2, 4 pixel) qualunque sia la roughness. Un riflesso di pochi pixel
  viene spalmato e il picco perde circa il 90%: è la causa del residuo di
  `roughness-custom` (93% dell'errore in 24 pixel) e dei bordi del pannello
  nello specchio.
- *Storia scartata a ogni frame nelle scene dinamiche.* La revisione di
  contenuto di ogni storia dipende dall'epoca globale del segnale, che cambia
  ogni volta che una qualunque istanza si muove. In `moving-light` e
  `disocclusion` il denoise `custom` lavora quindi solo nello spazio.
- *Cammini sole → sfera → sfera.* Anche col trasporto corretto, a 1 campione
  per frame sono estremamente sparsi: `rt8-control` resta a 0,51–0,61. Un
  filtro limitato al lobo da solo li lascia rumorosi (`roughness-custom` 0,571).
  Servono sia il limite di lobo sia un accumulo temporale più lungo dove la
  storia è valida.

**Peso di lobo nell'à-trous speculare (misurato, non ancora committato).**
Il filtro usa come guida l'asse di riflessione R = reflect(−V, N) e un
peso exp(−(1 − R_c·R_v)/α_c²), uguale nei tre kernel e nello specchio CPU,
con un nuovo test.

| Caso | Prima | Con il peso di lobo |
|---|---:|---:|
| `mirror-custom`, superficie dello specchio | 0,078 | **0,028** (passano tutte e tre le regioni) |
| `moving-light-custom` | 0,072 | 0,075 (ghost da 0,51 a 0,81) |
| `roughness-custom` | 0,178 | 0,571 |

È corretto dove la storia accumula (`mirror`, statica). Nelle scene dinamiche
la storia viene azzerata a ogni frame, quindi limitare il filtro lascia il
rumore. Sui cammini a coda pesante di `roughness` lo scambio è sfocatura
contro rumore. Va adottato solo insieme a due correzioni:
- storia valida anche quando si muove qualcosa altrove;
- regolarizzazione dichiarata più forte, α' = max(α_s, k·α_c).

Lo strumento di riferimento ha ora `--regularization-scale k` (default 1).

Scartata per assenza di beneficio misurato: media deterministica sul disco
solare al vertice secondario (`roughness` invariato, costo in più).

## 12. Punto di ripresa (2026-11-11, fine sessione)

**Branch.** `phase/f13-f14-closure` (worktree `.claude/worktrees/f13-f14-closure`),
tutto committato, nessun push. Build: `build/release` con
`-DPHOSPHOR_METALFX_DENOISED_FACTORY=ON -DPHOSPHOR_SHADER_DEBUG_INFO=OFF`
(come `closure-release` dello sviluppo), toolchain di sistema 32023.922.
Commit dopo il merge 2b285df: f30a93b, f3d728e, bb376ab, 5e94b3e, ae44691,
9db4a98, 7c2f285, 0a017af, fd6bf8b, 448bca7. Il branch `review/f13-reference-clips`
ha in più dd8e46c (`--regularization-scale`), non ancora unito alla chiusura.

**Chiuso con evidenza.**
- `mirror-rt8-control` passa su tutte le regioni.
- `ao-cavity-rtao` passa contro il riferimento AO indipendente.
- La fixture MetalFX nativa wide-HDR passa con D1.
- F14.1 «parametri estremi» passa: 2 casi, 4800 confronti, 0 errori.

**Aperto, in ordine di ripresa.**
1. **Clip `disocclusion`** (resumable): stesso comando di §7 con `--resume`,
   poi `build/f13-reference-clips-20261110/compare.sh disocclusion compare-disocclusion-fix`.
2. **Budget A/B/A** a macchina libera. Comando in
   `~/.codex/worktrees/f10-f12-development/phosphor/build/f13-f14-refinement-perf/protocol.json`
   (base + caso), binario di base `closure-release` contro `build/release`.
   Atteso circa −1,7 ms dalle verifiche; poi circa 4 ms da trovare nel denoise.
3. **Denoise speculare.** Patch del peso di lobo in
   `build/f13-reference-clips-20261110/patches/lobe-weight.patch`: `mirror`
   tutto verde, ma `moving-light` e `roughness` peggiorano da sola (§11). Serve:
   - storia speculare non azzerata dall'epoca di geometria;
   - regolarizzazione k > 1, con una nuova clip `roughness` resa con
     `--regularization-scale` e il motore allineato
     (`minRoughness = sqrt(k·max(r_c², 0.002))`);
   - per `moving-light`, invalidazione per pixel al posto del reset globale
     a ogni frame quando l'emettitore si muove.
4. **GTAO** (`ao-cavity-gtao` p95 0,081, poca occlusione sotto la sfera e sulle pareti).
5. **Non determinismo** sotto `MTL_SHADER_VALIDATION` + `--volume-oracle`
   (448bca7): da spiegare prima di dichiarare F14.4. Prossima prova
   `--frames-in-flight 1`.
6. **F14 mancanti:**
   - luna e stelle su GPU (righe nuove nell'oracolo numerico);
   - nebbia: slicing, luce mobile, disocclusione e costo;
   - nuvole: volo, controluce e bordo schermo;
   - F13.5 sonde e F13.3 ammissione nativa legata alla versione.
7. Roadmap ancora tutta non spuntata: nessun task F13/F14 è dichiarato chiuso.

## 13. Piano di ripresa completo (ordine di esecuzione)

Blocchi non elencati dall'altro agente ma necessari per chiudere: i test di
regressione dopo le modifiche al motore, F13.5 (sonde), F13.1 (fallback SSR
verso emettitori visibili), la vita del MetalFX nativo, l'esecuzione unica
dell'intero corpus, la roadmap, l'handoff e l'integrazione.

1. **Ripartenza.** Stato della clip `disocclusion` (riprendere con `--resume`),
   batteria, build `build/release` con le opzioni di §12.
2. **Rete di sicurezza sui 10 commit.**
   - `ctest`;
   - `tools/visual_check.sh` e le batterie F5–F8 (`f6_check.sh`, varianti, archivio);
   - matrice F9 60/60 sul nuovo binario.

   Prima di ogni altra modifica.
3. **Budget, misura di base.** A/B/A a macchina libera (comando in §12): effetto
   reale delle verifiche tolte e tempi per pass del denoise.
4. **Non determinismo** con validation e oracolo: `--frames-in-flight 1`,
   trovare la sincronizzazione mancante e correggerla. Può falsare le altre
   misure.
5. **Confronto `disocclusion`** nel runner. Completa la misura di F13.6.
6. **Denoise speculare (F13.4/F13.6)**, il blocco più grande:
   - storia non azzerata dall'epoca di geometria, con controlli per pixel sul
     colpo riflesso;
   - peso di lobo (patch salvata);
   - regolarizzazione dichiarata k > 1, con una nuova clip `roughness` (ore di
     CPU, a ritmo batteria) e il motore allineato;
   - invalidazione per pixel al posto del reset globale quando un emettitore
     si muove.

   Obiettivo: i quattro casi `*-custom` verdi.
7. **Costo del denoise:** circa 4 ms in meno, misurati in A/B/A, fino a 16,67 ms a 1080p nativo.
8. **F13.2 GTAO.** Qualità in spazio schermo; se il limite è intrinseco, va una
   decisione esplicita sulla soglia T0.
9. **F13.5 sonde.** Sonda cotta, riferimento, box parallax, confini dei volumi,
   variazione di roughness.
10. **F13.3.**
    - ammissione del nativo legata alla versione SDK;
    - caso `metalfx` contro la clip `roughness`;
    - ripetere la verifica della vita dello scaler.
11. **F14.4.** Righe luna e stelle nell'oracolo GPU; ripetibilità verde dopo il punto 4.
12. **F14.2 nebbia:** slicing, luce mobile, disocclusione e costo per volume.
    **F14.3 nuvole:** volo, controluce e bordo schermo, contro la coppia full-rate.
13. **F13.1.** Il fallback SSR non deve contare l'emissione già presa dalla DI.
14. **Chiusura.**
    - Esecuzione unica del corpus F13/F14 completo con il binario finale.
    - Matrice F9 e batterie di regressione.
    - `perf-log` e `opt-log`.
    - Spunta dei task verificati in roadmap.
    - Handoff per ID.
    - Integrazione e PR, da concordare col proprietario: i 12 commit dello
      sviluppo non sono mai stati pubblicati.

**Aggiornamento:** la clip `disocclusion` è completa (verificare `clip_report.json`); il passo 5 di §13 può partire con `compare.sh disocclusion compare-disocclusion-fix`.

**Verifica del lavoro di Codex (419fbb1, integrato con fast-forward in `phase/f13-f14-closure`).** ctest 12/12. Runner `night-and-regress`, 8 casi: atmo-zenith, atmo-horizon, atmo-space, atmo-extreme-dense/thin, daynight-a, atmo-night: 22500 confronti ciascuno (i 2400 di prima più 67 raggi celesti × 300 snapshot), 0 errori; `negative-f14-moon` fallisce su 300/300 snapshot come richiesto. Luna e stelle sono ora verificate su GPU; F14.4 resta aperto per il non determinismo (448bca7).

**Regressioni su HEAD (419fbb1), 2026-11-11 pomeriggio.** Binario di base: worktree `.claude/worktrees/f13-f14-baseline` a 2b285df, stessa toolchain 32023.922 e stesse opzioni. `tools/visual_check.sh` (API + shader validation, `BENCH8_ARGS=--instances 100000`): 8 bench su 8 identici pixel per pixel al binario di base, validation silenziosa. Matrice F9 su HEAD: 60/60, raggi ammessi dal criterio D3 2 su 2 808 814 come prima. Evidenze in `build/regress-20261111/`.

**Budget A/B/A (2026-11-11, Sponza 1080p, protocollo dello sviluppo, 3 ripetizioni).** Base 2b285df: GPU p50 22,02 / 22,10 ms (A/A2), p95 24,5 / 24,3 ms. HEAD 419fbb1: GPU p50 20,16 ms (19,86–20,48), p95 21,62 ms, frame p95 22,61 ms. Passate di verifica 1,66 ms → 0; denoise invariato a 8,8 ms. Allocazioni GPU 0. Mancano circa 5 ms al p95 per 16,67. Evidenze `build/regress-20261111/budget/`.

## 14. Denoise speculare e decisione D4 (2026-11-11, sera)

**dc6f3bb.** Tre modifiche:
- filtro à-trous speculare limitato dal lobo di riflessione;
- storia speculare che sopravvive al moto e viene invalidata per pixel
  (`DENOISE_SOFT_CHANGE`);
- opzione `--reflection-regularization-scale k`.

Esiti contro le clip indipendenti, solo casi `custom`:

| Caso | Prima | Ora |
|---|---|---|
| `mirror` | p95 superficie dello specchio 0,078 | **tutte le regioni verdi**: 0,028 / 0,025 / 0,024 |
| `disocclusion` | p95 0,070 | **tutti i gate verdi**: p95 0,057, ghost 0,016, recupero 1 frame |
| `moving-light` | p95 0,072 | p95 0,053 verde; ghost F8 rosso (vedi D4) |
| `roughness` (k = 1) | 0,178 | 0,571 |
| `roughness` (k = 4, motore e riferimento) | — | 0,113; il 93% dell'errore è in 24 pixel di riflessi sole → sfera → sfera, con un bias deterministico (motore 221 contro riferimento 153) da indagare |

**D4 (decisione del proprietario: ghost sull'errore che cambia), e5c0ec8.**
Coefficiente sull'intera clip: Σ(Δout·ΔD)/ΣΔD². Vale solo per le ROI F13;
F8 e F14 restano invariati, la soglia resta 0,25. Controlli sulla clip reale:

| Candidato | Valore | Esito |
|---|---:|---|
| riferimento ritardato di un frame | 0,95 | fallisce |
| accumulo esponenziale α = 0,16 | 0,46 | fallisce |
| `custom` | 0,000 | passa |
| `rt8` senza storia | 0,000 | passa |
| `custom` con storia azzerata | 0,005 | passa |

Un primo tentativo per singolo frame era dominato dal rumore: 1,9 anche
senza storia.

ROI congelate nuove: `build/f13-reference-clips-20261110/roi-config-d4.json`.
Clip `roughness` con k = 4: `roughness-k4/clip-k4`, 4096 campioni per seme.

**F14.4 ripetibilità (cd78c63).** La coppia `daynight-r1/r2` sugli shader di prodotto (sola API validation) è identica bit per bit su 300/300 frame e supera l oracolo (22500 confronti ciascuno). Il non determinismo compare solo con `MTL_SHADER_VALIDATION` e `--volume-oracle` insieme. È nel kernel della luce diretta con visibilità RT, a parità di ingressi; non dipende da frame in volo, timing seriale, aliasing o moto. Causa interna alla strumentazione di Metal; registrato come diagnostico.

**Budget, avanzamento (e31c9bf).** Nuovi default misurati: 2 iterazioni à-trous speculari, marcia SSR fino a 16 passi sulle superfici ruvide, RTAO a 2 raggi. Su Sponza 1080p, a macchina tranquilla: base p95 20,0 ms, con le tre misure p50 17,36 / p95 18,57 ms. Fixture invariate entro 0,0003 e ao-cavity-rtao p95 0,0101. Una conferma con i default è stata disturbata da una compilazione Rust concorrente e va ripetuta nell A/B/A finale. Mancano circa 1,9 ms al p95. Prossimi: test di cambiamento morbido basato sulla varianza in storia (patch `soft-variance-test.patch`, in verifica), compattazione della storia, riduzione a 1 campione.

## 15. Budget chiuso su M5 Max (2026-11-11, sera)

**Esito.** Sponza 1080p con percorso completo, protocollo del budget
(`--gpu-timing-serial`, 128 frame di riscaldamento e 256 misurati), binari
interleavati e stessa toolchain. Cinque esecuzioni pulite per binario:

| Binario | GPU p95 | Frame p95 |
|---|---:|---:|
| HEAD 42adefd | 14,91–15,39 ms | 15,74–16,36 ms |
| 954034b | 18,58–19,30 ms | 19,37–20,15 ms |
| base 2b285df | 22,4–23,1 ms | — |

Il gate di 16,67 ms è soddisfatto a 1080p nativo. Il margine è circa 1,3 ms
sulla GPU e 0,3–0,9 ms sul frame. Una finestra disturbata, in cui tutti i
binari sono più lenti di 2–4 ms, è riportata ed esclusa
(`docs/perf-log.md`).

**Commit del secondo e terzo giro**, ognuno verificato come indicato nel suo
messaggio:

| Commit | Contenuto | Verifica |
|---|---|---|
| 96d1507 | un campione di riflesso scrive il record ridotto; via il passo di riduzione (0,39 ms) | bit-identico |
| 5cd7aab | storia del denoise compatta: intestazione per buffer, record da 64 B, estensione speculare da 16 B (0,55 ms) | bit-identico; negativi ancora rossi |
| 71bfed3 | irradianza DDGI non filtrata, `--ddgi-denoise on\|off` (1,66 ms) | fixture invariate alla quarta cifra |
| 42adefd | AO a 2 stride, `--ao-atrous-iterations` (1,1 ms) | ao-cavity-rtao p95 0,0101 → 0,0070 |
| 10b68a8 | correzione: cattura lineare senza `--debug-lighting` | — |

L'irradianza DDGI è un'interpolazione deterministica delle sonde. Su F12
Cornell il valore grezzo era già migliore del filtrato: NRMSE 23,31% contro
24,19%. Per riprodurre il caso congelato F12 «ddgi filtrato» serve
`--ddgi-denoise on`.

**Regressioni** su 42adefd:
- `visual_check` 8/8 identico pixel per pixel alla base, con
  `BENCH8_ARGS="--instances 100000"` come per il riferimento: senza
  quell'argomento il bench 8 disegna 1M di istanze e differisce del 97%;
- ctest 12/12;
- runner: cicli di vita a 2 e 4 viste verdi; `negative-f13-history`,
  `-motion` e `-normal` falliscono come richiesto.

**Scoperta.** `MTL_SHADER_VALIDATION` rende non deterministiche anche le
fixture di riflessione RT: HEAD contro sé stesso differisce in 299 frame su
300, fino all'1,8% relativo. Non succede solo con `--volume-oracle`. I
confronti bit per bit vanno fatti con la sola API validation.

**Restano aperti**, nell'ordine di §13:
- `roughness` (bias deterministico sole → sfera → sfera, k = 4);
- F13.2 GTAO (p95 0,081, un bias);
- F13.5 sonde;
- F13.3: ammissione del nativo e vita dello scaler;
- F14.2 nebbia e F14.3 nuvole: i casi mancanti;
- F13.1: doppio conteggio del fallback SSR;
- corpus completo, roadmap, handoff, PR.

## 16. `roughness`, accumulo stazionario e aumento dei campioni (2026-11-11, notte)

**Diagnosi rifatta.** Le coordinate del brief (194,259) erano con le righe
capovolte del PFM. Mediato su 300 frame, il motore coincide col riferimento
k = 4: errore assoluto medio 0,0116, e grezzo, rt8 e custom hanno la stessa
media. L'errore misurato dal gate è rumore Monte Carlo:
- il pixel (259,194), riflesso sole → sfera 0,04 → sfera 0,1 da 131 unità
  (pavimento al sole circa 2), produce il 53,6% dell'errore quadratico della
  clip; i primi 5 pixel ne producono il 77,6%;
- a 1 campione la sua deviazione standard per frame è 143; con l'EMA a
  α = 0,08 scende a 31;
- anche il controllo rt8 (8 campioni, senza denoise) fallisce: p95 0,167.

**c0527d6, accumulo stazionario.** Se né il contenuto né la camera della
vista cambiano, la storia speculare diventa una media corrente fino a 128
campioni. `roughness` k = 4 passa da 0,1135 a 0,0707; dal frame 20 in poi
vale 0,048. Resta solo l'avvio da storia vuota, circa 20 frame, contro i 4
del recupero F8. Lo specchio dimezza il flicker; luce mobile e disocclusione
non cambiano (non sono mai stazionarie). Sponza, sempre in cambio morbido,
resta bit-identico.

**ee21dce, aumento dopo un reset.** Nei primi 8 frame di una storia
speculare, i pixel sotto roughness 0,2 usano 8 campioni RT ridotti nel
kernel. Il record porta il numero di campioni e la media stazionaria è
pesata per campioni: un aumento pesato per frame non serviva, la lucciola
successiva rientrava con peso 1/7.

| fixture | p95 prima | p95 dopo | max prima | max dopo |
|---|---|---|---|---|
| specchio | 0,0246 | 0,0186 | 0,072 | 0,031 |
| luce mobile | 0,0531 | 0,0516 | 0,0732 | 0,0568 |
| disocclusione | 0,0573 | 0,0567 | 0,0777 | 0,0612 |
| `roughness` k = 4 | 0,0707 | 0,0439 / 0,0639 / 0,0630 (semi 1–3) | — | — |

Nella disocclusione il recupero dopo il taglio scende a 0 frame. Su
`roughness` il risultato dipende dal seme: non è un pass robusto.

Costo su Sponza:
- aumento ristretto (roughness < 0,2): GPU p95 invariato, p99 +0,5 ms;
- aumento su tutta la banda RT (roughness < 0,4): p95 15,4 → 16,5 ms,
  p99 20 ms. Scartato.

**`roughness` con k = 1 (default):** p95 0,204 (era 0,571). La fixture
richiede una regolarizzazione dichiarata k > 1. È la decisione aperta D5:
clip k = 8 e k = 16 in rendering in `roughness-k8/` e `roughness-k16/`.

**F13.1.** Il residuo risolto contiene l'emissione (`material_shading.h`) e
il fallback SSR la riflette. La DI tiene già la quota smoothstep(0,2; 0,4)
dello speculare degli emettitori triangolari, quindi un emettitore visto in
SSR da un ricevitore sopra roughness 0,2 è contato due volte. Nessuna
fixture lo esercita: su `roughness` il pannello non è visibile all'SSR e le
sfere ruvide sono anzi più scure del riferimento (−3,5% a 0,5, −5,5% a 0,9:
il fallback a sonda non contiene le interriflessioni). Sponza non ha
emettitori. Per chiuderlo servono:
- una fixture «ricevitore ruvido + emettitore visibile» con il suo
  riferimento;
- un'emissione separata dal resolve, sottratta dalla sorgente SSR.

**F13.3.** `permitsNativeEncoding` ammette il nativo solo per fixture
diagnostiche controllate. L'ammissione HDR richiede le prove di qualità del
nativo sulle fixture F13 (caso `metalfx-native` contro le clip), oltre alla
fixture wide-HDR che passa D1. Non è iniziata.

**D5, dati (clip k = 8 e k = 16 convergenti, 4096 campioni per seme).** Il
motore, con lo stesso k del riferimento e i default attuali, sui semi di
illuminazione 1–3:

| k | p95 `roughness-custom` | max | esito |
|---|---|---|---|
| 1 | 0,204 | 0,486 | fallisce |
| 4 | 0,044 / 0,064 / 0,063 | ≤ 0,097 | dipende dal seme |
| 8 | 0,031 / 0,039 / 0,033 | ≤ 0,071 | passa, margine 35–48% |
| 16 | 0,030 / 0,031 / 0,030 | ≤ 0,073 | passa |

Con k = 8, `mirror`, `moving-light` e `disocclusion` hanno metriche identiche
a k = 1 alla quarta cifra. In quelle scene vale α_B ≥ k·α_A su tutti i
percorsi rilevanti, quindi la regola non agisce. Raccomandazione: k = 8,
il valore più piccolo che passa in modo robusto.

## 17. F13.2 GTAO: era un difetto, non un limite dello spazio schermo (2026-11-11, notte)

La mappa d'errore mostrava che il GTAO non vedeva gli occlusori sul
pavimento attorno a sfera e muri, ben visibili sullo schermo, e produceva
bande sulle loro facce. Le varianti (senza attenuazione, senza erosione,
16 direzioni × 32 passi) restavano tutte a p95 0,074–0,080.

Ricerca con diagnostiche per pixel:
- ricostruzione della profondità esatta (errore < 0,1 mm);
- `v` allineato al raggio di vista (coseno 1);
- i 64 campioni di un pixel cadono su una sola linea verticale;
- `screen.y` = −2,8·10⁸ px a parità di NDC corrette.

Causa: `float2(p.width*.5f,-p.height*.5f)`. `p.height` è senza segno, quindi
`-p.height` si avvolge a 2³²−360 prima della conversione e ogni fetta
collassa su un asse dello schermo.

Correzione (34ca3e2): `-float(p.height)*.5f`. `ao-cavity-gtao` contro la clip
AO indipendente passa da p95 0,0807 a 0,0370, max 0,037, flicker 0,0001:
tutti i gate verdi. Lo stesso schema in `temporal_motion.h` agisce su un
float ed è corretto.

## 18. F14.2 e F14.3: i casi di verifica (2026-11-12, questa chat)

Il compito `CODEX-TASK-F14.2-F14.3-CASES-2026-11-11.md` è stato eseguito
qui, sul branch `phase/f13-f14-closure`. I commit:
- ef54033, F14.3: storia delle nubi legata solo alle discontinuità;
- 79777cb, F14.2/F14.3: casi, controlli e negativi.

Nessun push. Giro finale: `build/f14-cases-20261112/final`, 48 casi, diff
del worktree sha256 1e5cb859 (identico al commit). Per ruolo:
- numerici 14/14;
- controlli 13/13;
- funzionali 5/5;
- ripetibilità 1/1;
- coppie a piena frequenza 13/15.

Le due coppie che falliscono sono `clouds-flight-top` HDR e T, descritte
sotto. ctest 12/12.

**Regressione.** I casi F14 esistenti mantengono i loro conteggi:
- 22500 confronti per caso, 23400 per `fog-homogeneous-oracle`;
- `volume-history-static-positive` con storia esercitata su 299/300
  snapshot;
- tutti i `negative-f14-*` falliscono come prima.

`clouds-reconstructed` cambia per scelta (ef54033): ora riusa la storia
anche con l'orologio in corsa, 6,41 M riusi, e passa con p95 0,0010.

### F14.2 nebbia froxel

**Slicing (`fog-slicing`, `fog-slicing-height`, ruolo numerico).** La
fixture `fog-depths` (pareti da 0,7 m a 220 m, pavimento di 2 km) e una
sonda GPU confrontano ogni pixel con il mezzo omogeneo analitico, cioè
trasmittanza exp(−σd) e in-scattering, per fasce di profondità. Ogni fascia
ha almeno 9604 campioni. L'errore di discretizzazione delle fette è
riportato a parte dall'errore di implementazione:

| caso | discretizzazione max (gate 1%) | fascia peggiore | implementazione max / tolleranza |
|---|---|---|---|
| omogeneo | 0,31% | 25–125 m | 0,0045 |
| altezza, falloff 0,05/m | 0,69% | 125 m–far | 0,0186 |

**Luce mobile (`fog-moving-light`).** Emettitore con gradini di colore,
sole veloce e salti dell'orologio a 60/120/180/240. Il controllo è senza
storia, con 16 campioni locali per froxel. Risultati:
- p95 0,0031, max 0,0034, flicker 0,0010, ghost support 0,0019;
- recupero 0 frame su tutti gli eventi.

Riuso della storia: 0. La firma del volume contiene la luce, quindi con la
luce in moto la storia froxel è sempre invalidata. Il caso dimostra che non
restano scie per costruzione, non la qualità di una storia riusata sotto
moto. Il negativo `negative-f14-fog-stale` conserva la storia quando la luce
si muove e fallisce:
- p95 0,505, max 0,839;
- ghost support 2,0·10⁴;
- nessun recupero.

**Disocclusione (`fog-disocclusion`).** Camera in traslazione con
l'orologio fermo, così la storia è ammessa; l'occlusore scivola tra 2 e 4 s,
con un taglio di camera a 4 s. Risultati:
- 13,2 M riusi;
- p95 0,0084, max 0,0094;
- ghost support 0,185 (gate 0,25);
- recupero dopo il taglio 0 frame.

Residuo dichiarato: lo scivolamento dell'occlusore non è un evento di
recupero dichiarato. È coperto da p95, max e ghost sull'intera sequenza, non
da un conteggio di frame.

**Costo (`fog-cost`, misura e non gate; `docs/perf-log.md`, 2026-11-12).**
1080p nativo, timing seriale, 128 + 256 frame, A/B/A × 3, macchina quieta
(render delle clip sospeso), 0 allocazioni GPU. Su Sponza la sorgente froxel
con visibilità RT è bimodale:
- frame economici circa 0,1 ms, frame costosi 1,5–2,2 ms;
- default p50 0,105, p95 1,57;
- 120 m: sempre economica, 0,12;
- 2000 m e griglia 128×72×64: sempre costosa, 1,42 e 2,01.

Al default i frame costosi spostano il p95 GPU del frame di circa 1 ms
(16,23 contro 15,21 a 120 m). Altri costi:
- composizione 0,09 ms, 0,18 a 8× celle;
- campioni locali circa 0,017 ms per campione in più a 64×36×32.

Quale posa o popolazione di froxel inneschi lo stato costoso non è
isolato: il report non ha tempi per frame. È un candidato, non lavoro
attivato.

### F14.3 nubi

**Scelte di progetto emerse nei casi.**
- ef54033: la storia delle nubi aveva come epoca la firma di tutti i volumi,
  che cambia a ogni frame con il sole o il vento in moto. La ricostruzione
  non riusava mai la storia. L'epoca nuova contiene solo le discontinuità:
  scena, epoca dell'orologio, revisione dei parametri fisici, generazione
  della pipeline. Il moto continuo resta al passo temporale (avvezione del
  vento, rifiuti di posizione, profondità e vicinato).
- Il volo originale a 700 m/s era un taglio di camera a ogni frame: oltre
  4 m per frame, quindi 240 m/s a 60 fps, il motore azzera la storia. Ora ci
  sono due voli a 200 m/s (3,3 m per frame): `flight-base` attraversa la base
  a 1500 m e `flight-top` il tetto a 4000 m, con un controllo della quota per
  frame. **Limite di prodotto:** sopra i 240 m/s la ricostruzione delle nubi
  non ha mai storia.
- Una regola di rifiuto sul moto relativo della camera è stata provata e
  rimossa. Non scattava mai, e il ghost di `flight-top` non era lag: un
  controllo senza storia dà support 0,618 (HDR) / 0,395 (T).

**Esiti** (ricostruita contro `--cloud-full-rate`; HDR a scala 25 e
trasmittanza composta T a scala 1; copertura T < 0,5 per frame, limite
0,02):

| caso | p95 HDR | p95 T | support max HDR / T | copertura Δ max | riusi |
|---|---|---|---|---|---|
| flight-base | 0,0008 | 0,0235 | 0,093 / 0,026 | 0,0030 | 6,13 M |
| flight-top | 0,0011 | 0,0276 | **0,292 / 0,402** | 0,0030 | 3,38 M |
| backlit (3 ROI) | ≤ 0,0014 | ≤ 0,0204 | ≤ 0,034 / ≤ 0,005 | 0,0043 | 7,82 M |
| screen-edge (4 ROI) | ≤ 0,0013 | ≤ 0,0149 | ≤ 0,038 / ≤ 0,020 | 0,0093 | 8,82 M |
| screen-edge-deck (4 ROI) | ≤ 0,0009 | ≤ 0,0302 | ≤ 0,065 / ≤ 0,075 | 0,0128 | 5,14 M |

`screen-edge-deck` guarda dall'alto uno strato di nubi a 5000 m, con
pitch −20° e yaw 60°/s. Ogni frame entra dal bordo destro una fascia
dell'1,75% della larghezza, nuova e senza storia: è la ROI
`entering-frame`.

**`flight-top` fallisce solo il gate ghost support** (0,292 / 0,402 > 0,25).
Gli altri gate passano: p95, max, flicker, recupero e copertura. Diagnosi:
- è una distorsione spaziale statica nella fascia dell'orizzonte, correlata
  in segno con l'andamento monotono del riferimento durante la salita;
- la metrica support la legge come «ghost» anche senza storia (controllo
  sopra);
- il riferimento converge: a piena frequenza, 128 contro 512 passi di
  marcia danno support 0,052;
- con la metrica D4 (errore che cambia, già adottata per F13.6) il caso
  passa: 0,0073 / 0,0126 sulla sequenza, 0,112 / 0,095 sul frame peggiore.

Il gate non è stato toccato.

**Negativo `negative-f14-cloud-edge`.** La storia riproiettata fuori dallo
schermo si avvolge sul bordo opposto e salta ogni rifiuto (limiti, posizione,
profondità, clamp di vicinato). Fallisce sulle ROI di ingresso:
- `entering-frame`: p95 0,260, support 2,10, flicker 0,228;
- `entering-band`: p95 0,188.

Passano `full-frame` e `leaving-edge`: il difetto è locale al bordo, come
deve. Prima versione scartata: togliere soltanto il test dei limiti non
fallisce. Il test di posizione da 100 m rifiuta comunque la storia sbagliata
(screen-edge e deck, giri 3–4).

### Decisione D6 del proprietario (2026-11-12): D4 vale anche per F14

Il proprietario ha risposto «Sì, D4 anche F14». Tutte le ROI del runner,
F13 e F14, misurano ora il ghost con la metrica «errore che cambia»;
la frazione support resta nei report come diagnostico (commit 9fa928a).

Giro congelato `build/f14-cases-20261112/d6`, 48 casi F14, stesso binario
del giro finale; lo `source_sha256` registrato coincide con l'albero del
commit:

| ruolo | prima (final) | dopo (d6) |
|---|---|---|
| numerical | 14/14 | 14/14 |
| control | 13/13 | 13/13 |
| full-rate-pair | 13/15 | 15/15 |
| functional | 5/5 | 5/5 |
| repeatability-pair | 1/1 | 1/1 |

Sono cambiati solo i due casi `flight-top` (HDR e trasmittanza): errore
che cambia 0,0073 e 0,0126 contro la soglia 0,25. I negativi falliscono
ancora:
- `negative-f14-fog-stale`: p95 0,505, frame peggiore, recupero e ghost
  (errore che cambia 0,254);
- `negative-f14-cloud-edge`: p95, frame peggiore e flicker nella fascia
  che entra. Il suo gate ghost non scatta più (0,112 e 0,204): la metrica
  D4 è meno sensibile della support su questo negativo, che resta
  coperto dagli altri tre gate.

### Residuo per ID

- **F14.2:**
  - densità costante e slicing: verificati;
  - luce mobile: verificata senza riuso di storia (per progetto);
  - disocclusione: verificata, lo scivolamento non è un evento di recupero
    dichiarato;
  - costo: misurato (perf-log), con la bimodalità della sorgente RT non
    spiegata per frame.
- **F14.3:**
  - volo: base e top verdi (top con D6);
  - controluce, bordo schermo e copertura: verificati;
  - negativo di bordo: verificato;
  - limite dei 240 m/s documentato.
- Nessuna soglia è stata allargata. La fase non è accettata finché restano
  aperti i punti F13 (F13.1 doppio conteggio SSR, F13.3 ammissione HDR
  nativa, F13.5).

### Clip k = 8 per F13.6 (D5)

`moving-light` con la clip indipendente rifatta a k = 8, renderizzata fino a
6144 campioni per seme (`moving-light-k8/clip-k8`). `moving-light-custom`,
con le ROI D4:

| | p95 | max | flicker | bias | recupero |
|---|---|---|---|---|---|
| clip k = 8 | 0,0513 | 0,0565 | 0,0021 | −0,0026 | 0 frame |
| clip k = 1 (d5) | 0,0516 | 0,0568 | 0,0021 | −0,0036 | 0 frame |

Tutti i gate sono verdi. Il risultato conferma che in questa scena la
regola k non agisce.

`disocclusion` con la clip k = 8 (`disocclusion-k8/clip-k8`, 240/240
gruppi convergenti, 4096 campioni per seme; confronto
`disocclusion-k8-cmp`, ROI D4), `disocclusion-custom`:

| | p95 | max | flicker | ghost D4 | bias | recupero al taglio |
|---|---|---|---|---|---|---|
| clip k = 8 | 0,0565 | 0,0609 | 0,0028 | 0,000 | −0,0067 | 0 frame |
| clip k = 1 (d5) | 0,0567 | 0,0612 | 0,0028 | 0,000 | −0,0074 | 0 frame |

Tutti i gate sono verdi con entrambe le clip: anche qui k = 8 non cambia
l'esito. D5 (k = 8) è quindi coperta su `roughness`, `moving-light` e
`disocclusion`.

## 19. F13.1: emettitori contati una volta sola nei fallback (2026-11-12, questa chat)

Commit `bae13c6` (locale, nessun push).

**Il difetto.** La DI primaria integra il lobo speculare di ogni emettitore
geometrico per la quota che i riflessi RT non coprono:
- con i riflessi RT attivi, `smoothstep(0,2; 0,4; roughness)`;
- senza RT, tutta.

Il fallback (SSR, cache di radianza, sonda catturata) viene scelto proprio
con quella probabilità. Però leggeva l'immagine risolta con l'emissione
dentro, quindi un emettitore visto dal fallback contava due volte.

**La correzione.**
- La risoluzione porta il proprio termine emissivo con ogni ricevitore DI,
  in `GPUDISurface::pad[1]`. Il formato è RGB a esponente condiviso
  (`renderer/emission_pack.h`): esatto per le emissioni delle fixture,
  mai sopra il valore di ingresso.
- La sorgente SSR lo sottrae nel pre-compose.
- Il percorso cache sottrae l'emissione del ricevitore colpito.
- Le catture delle sonde (RT e raster) la omettono.
- Tutto vale solo quando gira una DI non legacy. I raggi RT dei riflessi
  tengono l'emissione del punto colpito.

**Fixture `rough-emitter`** (statica):
- due strisce di pavimento metallico, roughness 0,3 (banda mista) e 0,5
  (solo fallback);
- un pannello emissivo bianco (Le 2,5) visibile allo schermo;
- sole spento.

Le ROI sono i riflessi del pannello sulle due strisce, poste dalla
geometria. Scala 2,0 = F0·Le = 2,25 arrotondato per difetto. Tutto è stato
dichiarato prima di leggere un errore.

Casi:
- `rough-emitter-{raw,custom,rt8-control}` in modalità RT;
- `rough-emitter-ssr-custom` in modalità SSR, con DI brute.

Correzione di dichiarazione fatta durante il lavoro: le ROI
personalizzate imponevano `require_change` anche a questa scena statica.
Ora è falso come per `mirror` e `roughness`; le ROI delle nubi restano
invariate (test).

Clip indipendente `rough-emitter/clip`, k = 8: converge a 12288 campioni per
seme (NRMSE fra semi 1,3% e 1,7%). A 6144 campioni non convergeva (2,3%) ed
è conservata come `clip-6144-unconverged`.

| caso | ROI | prima: p95 / bias | dopo: p95 / max / bias |
|---|---|---|---|
| `rough-emitter-custom` | r 0,3 | 0,129 / +0,070 | 0,032 / 0,132 / −0,007 |
| `rough-emitter-custom` | r 0,5 | 0,127 / +0,077 | 0,010 / 0,028 / −0,002 |
| `rough-emitter-ssr-custom` | r 0,3 | 0,260 / +0,143 | 0,032 / 0,130 / −0,007 |
| `rough-emitter-ssr-custom` | r 0,5 | 0,128 / +0,077 | 0,010 / 0,028 / −0,002 |
| `rough-emitter-raw` (controllo) | r 0,3 / 0,5 | bias +0,077 / +0,083 | bias +0,0001 / −0,0000 |

Il «prima» conferma la diagnosi in modo quantitativo. A roughness 0,3 la
modalità SSR (fallback sempre) ha bias doppio rispetto a RT (fallback metà
delle volte); a 0,5 le due modalità coincidono. Il max a r 0,3 (0,132
contro 0,15) è il margine più stretto: viene dai primi frame. Il binario
«prima» è `af5910ec`, conservato in `build/prefix-f13.1`.

Confronti:
- `compare-emitter-before`;
- `compare-emitter-fix1`;
- regressione `*-f13.1-cmp`.

**Regressione** contro le clip k = 8, casi `-custom`, tutti verdi:

| scena | p95 prima | p95 dopo |
|---|---|---|
| `mirror` (floor-front) | 0,0237 | 0,0224 |
| `roughness` | 0,0309 | 0,0308 |
| `moving-light` | 0,0513 | 0,0513 |
| `disocclusion` | 0,0565 | 0,0565 |

ctest 12/12, doctest 688 casi (nuovo `test_emission_pack.cpp`).

**Residuo F13.1:**
- I ricevitori disegnati dal fallback indicizzato di overflow non portano
  l'emissione (`pad[1] = 0`), quindi lì il doppio conteggio resta.
- L'esclusione nelle catture delle sonde compila, ma nessun caso esegue la
  DI non legacy con una sonda catturata: non è verificata.
- Il corpus completo F13/F14 e `visual_check` vanno rifatti col binario
  finale (punto di chiusura). Qui sono stati rieseguiti solo i casi F13.6
  contro le clip.

## 20. F13.3: il nativo MetalFX misurato sulla fixture `roughness` (2026-11-12, questa chat)

Commit `9b6b523` (locale). Nuovo flag `--metalfx-native-diagnostic`: fa
girare lo scaler denoised nativo sugli input reali del motore, nel dominio
`ControlledFixtureDiagnostic`. Misura e basta: il report resta
`denoise_native_production_qualified: false`.

Spike contro la clip indipendente `roughness` k = 8, con la ROI e i limiti
congelati di `roughness-custom` (HDR lineare, 640×360):

| esecuzione | frame valutati | p95 | max | flicker | bias |
|---|---|---|---|---|---|
| nativo, 300 frame (nativo dal 208, per la compilazione del gateway) | 208–299 | 0,164 | 0,199 | 0,035 | +0,003 |
| custom, stessi frame | 208–299 | 0,029 | 0,029 | 0,0003 | +0,003 |
| nativo, 900 frame (893 nativi) | 600–899 | 0,174 | 0,731 | 0,035 | +0,003 |

Limiti: p95 0,06, max 0,15, flicker 0,02. Il nativo li supera tutti e tre.
Non converge (media dei primi e degli ultimi 50 frame 0,121 / 0,135) e il
residuo non ha bias: è rumore o tremolio rimasto, non energia sbagliata.
L'ammissione HDR del nativo resta quindi **negata con evidenza**, e la
produzione continua a scegliere il custom (`UnqualifiedRadiometricDomain`).

**Vita dello scaler** sull'immagine MetalFX attuale:
- `METALFX-DENOISED-LIFETIME weak_nil=1 timing_reclaimed=1 signature_matched=1`;
- `leaks --atExit`: 0 leak (240 frame).

Dati in `build/f13-reference-clips-20261110/spike-f13.3-native{,-900}`.

**Cosa resta di F13.3 e decisione richiesta (D7).** L'adapter verso i canali
di F7.6 c'è e funziona. La vita dello scaler è verificata. Il nativo non
raggiunge la qualità F13.6 del riferimento indipendente. Il task in roadmap
(«MetalFX denoised upscaler collegato ai canali di F7.6») non è soddisfatto
come prodotto. Le strade possibili:
- lasciare F13.3 aperto (parziale) finché un'altra versione di MetalFX non
  passa lo stesso spike;
- ridefinirne lo scopo: adapter più fallback custom misurato, con il nativo
  respinto per qualità.

Non ho tentato di tarare il nativo (esposizione, guide): uno spike a tempo
limitato, il risultato è lontano dal limite (2,9×).

**Decisione D7 del proprietario (2026-11-12): «Ridefinisci lo scopo».**
F13.3 è ora l'adapter verso i canali di F7.6, con le fixture SDK reali, il
criterio D1, la vita dello scaler verificata e il nativo misurato e respinto
per qualità. La produzione usa il custom. Roadmap e policy radiometrica sono
aggiornate (commit sul branch). La spunta arriva dopo l'esecuzione finale del
corpus.

## 21. F13.5 sonde statiche e seguito di F13.1 (2026-11-12, questa chat)

Commit locali:
- `be4ee1b`: box della sonda e pesi;
- `d9f1c82`: proprietà a bande degli emettitori;
- `609d124`: tabella DFG.

**Fixture `probe-box`:**
- una stanza chiusa 8 × 4 × 16 m di pareti nere emissive, un colore per faccia;
- quattro strisce di pavimento metallico (roughness 0,04 / 0,3 / 0,6 / 0,9).

Le pareti hanno radianza esatta con qualsiasi modello di luce e la stanza
coincide col box della sonda. Clip indipendente `probe-box/clip`: 8192
campioni per seme, NRMSE fra semi ≤ 1,5%. ROI controllate sulle posizioni
nel mondo esportate: ogni pixel è sul pavimento, dentro la sua striscia con
0,1 m di margine. Scala 1,8 = F0 · Le della parete più luminosa.

Casi:
- `probe-box-t0`: preset T0, senza RT, cattura raster, luce legacy;
- `negative-f13-probe-parallax`: `--debug-probe-no-parallax`, deve fallire
  contro la clip (nuovo gating `expected_clip_failure`);
- `probe-box-rt-capture`: DI brute più cattura RT.

**Tre difetti trovati e corretti:**
1. Il box di parallasse era l'unione delle sfere di culling delle mesh: per
   la stanza veniva circa 31 × 27 × 27 m. Ora è l'AABB esatto della
   geometria (`GpuScene::meshLocalBounds`).
2. Il peso calava verso l'ambiente nell'ultimo metro *dentro* il box, quindi
   pavimenti e pareti, che stanno sulle sue facce, ricevevano solo una
   frazione della sonda. Ora il peso è 1 dentro il box, facce comprese, e la
   sfumatura avviene fuori.
3. Lo split-sum usava l'approssimazione analitica mobile del DFG, il 14% sotto
   il BRDF del motore a roughness 0,3–0,6. Ora usa una tabella 64 × 64
   integrata dal BRDF esatto (`tools/dfg_table.py`). Nei test la tabella
   concorda con due riferimenti indipendenti entro 0,0059.

`probe-box-t0`, p95 per striscia:

| striscia | prima | box | box + DFG |
|---|---|---|---|
| r 0,04 | 0,245 | 0,019 | 0,008 |
| r 0,3 | 0,086 | 0,080 | 0,031 |
| r 0,6 | 0,073 | 0,079 | 0,042 |
| r 0,9 | 0,054 | 0,058 | 0,039 |

Il negativo senza parallasse fallisce (0,414 a r 0,04). Questo verifica
parallasse, sweep di roughness e assenza di dipendenza da RT nel preset
base. I confini dei volumi sono coperti così: le facce del box nel caso
GPU, la sfumatura esterna solo nei test del mirror CPU.

**Seguito di F13.1 (`d9f1c82`).** `probe-box-rt-capture` ha mostrato che,
senza riflessi RT, bae13c6 dava alla DI tutto il lobo degli emettitori. La
DI lo stima con un campione di area per luce: su un lobo quasi a specchio il
denoiser taglia i rari valori enormi, e a r 0,04 mancava metà del riflesso
(p95 0,60, bias −0,42). Ora la DI tiene la banda `smoothstep(0,2; 0,4)` in
ogni modalità in cui un riflesso vede emettitori (RT o il suo fallback SSR,
SSR, sonda catturata). I fallback aggiungono la quota restante:
- SSR dall'emissione impacchettata;
- cache sottraendo il complemento;
- sonda con la luminanza emissiva catturata nell'alfa.

Un test CPU verifica che le quote sommino a uno in ogni modalità.
`rough-emitter` resta verde (custom 0,032/0,010, SSR 0,030/0,010).

`probe-box-rt-capture`, p95 dopo i tre commit: 0,0095 / 0,024 / 0,028 /
**0,126** (r 0,9 fallisce, bias +0,11).

**Aperto: sonda e orizzonte del ricevitore.** La sonda catturata in RT
contiene il pavimento illuminato. Il lobo prefiltrato di un ricevitore ruvido
lo raccoglie sotto il proprio orizzonte: un pavimento piano che riflette sé
stesso. Nel caso T0 il pavimento catturato è scuro, perché senza luci la
cattura raster non lo illumina. In una scena T0 con luci lo stesso limite
può comparire. È un limite delle sonde prefiltrate al centro: non l'ho
trattato, e il caso resta di gating e rosso.

## 22. Chiusura: corpus completo, casi mancanti e residuo dell'overflow (2026-11-12, questa chat)

**Primo giro completo** (`build/f13-f14-final-20261112` del worktree, 83 casi,
binario sha256 b4e36020 = 609d124, ROI `roi-config-d4.json`):
- control 27/27, full-rate-pair 15/15, functional 13/13, numerical 14/14,
  repeatability 1/1;
- independent-clip 10/13. I tre mancanti:
  - `probe-box-rt-capture` rosso a r 0,9 (p95 0,126), il limite noto di §21;
  - `metalfx-fallback` senza riferimento;
  - `probe-parallax-cooked` non eseguito, perché manca l'ingresso esterno
    (`--probe-dir`). Per questo il runner dice «functional FAIL».

I valori per clip coincidono con le misure per fixture: mirror 0,018/0,018/0,022,
roughness 0,031, moving-light 0,051, disocclusion 0,057, rough-emitter
0,032/0,010 (SSR 0,030/0,010), probe-box-t0 0,008/0,031/0,042/0,039,
ao-cavity gtao 0,037 e rtao 0,005. Il negativo senza parallasse fallisce
(0,414). Tabella: `independent-table.txt`.

**`metalfx-fallback` contro la clip `roughness`.** Stessa scena e stessa camera
di `roughness-custom`: gli do la stessa ROI congelata, copiata senza
modifiche (`roi-config-d4-fallback.json`). p95 0,0309, max 0,071, flicker
0,0004: passa come il custom, perché il fallback sceglie davvero il custom.
Un primo confronto senza config ROI usava regione e scala predefinite (p95
0,0617 anche per il custom) e non è valido.

**`probe-parallax-cooked`.** Nuovo `tools/probe_pfm_author.py` (60f2659). Scrive
le sei facce PFM dalla tabella delle cube map di Metal, senza usare gli
helper del motore. Il pattern è il gradiente della sonda analitica interna:
- cooked contro analitica, frame 63: max |d| 9,5·10⁻⁷, 0 canali su 691 200
  sopra 10⁻⁴;
- negativo con le righe salvate dall'alto: 0,077 su 334 267 canali;
- runner con `--probe-dir`: funzionale PASS, `probe_source cooked-linear-pfm`.

**Residuo F13.1 dell'overflow, corretto (75c0b6e).** I ricevitori del fallback
indicizzato avevano `pad[1] = 0`, quindi la sorgente SSR teneva l'emissione
che la DI integra già. Misura con `tools/f13_overflow_emission_check.sh`:
`rough-emitter` SSR con ogni frame forzato nell'overflow da
`--debug-meshlets-corrupt count`, confrontato col percorso visibility.

| | r 0,3 | r 0,5 |
|---|---|---|
| prima | +0,149 | +0,161 |
| dopo | −0,00004 | −0,00002 |

Massimo 0,0044. L'immagine del percorso visibility è identica bit per bit al
binario precedente. L'archivio riceve il descrittore registrato dal
compilatore (cinque attachment) al posto dei due vecchi; miss 0. Un harvest
completo rietichetta ogni funzione e duplica il corpus (render 171 → 231),
per questo l'aggiornamento è mirato. ctest 13/13.

Un primo tentativo aveva un commento a metà riga che disattivava
`out.valid`: pavimento nero sul percorso di overflow. L'ho trovato per
bisezione e corretto prima del commit.

**`visual_check` (API + shader validation, `BENCH8_ARGS=--instances 100000`).**
Riferimenti presi da 42adefd. Su 609d124: 8/8 bench identici pixel per pixel,
e lo stesso con `--debug-graph-transients`, `--debug-split-encoding` e
`--debug-async-compute`.
