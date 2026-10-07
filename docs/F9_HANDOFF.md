# F9 — Consegna per integrazione

2026-10-07. **DEVELOPMENT_ACCEPTED sul M5 Max 128 GB**, macOS 27.2
`26B5091g`; PR [#18](https://github.com/yabanana/phosphor/pull/18) per CI
e merge. Il risultato riguarda F9.1–F9.5. F9.6 resta candidato non attivato.
La certificazione su altri chip, compreso M3/T0, è esterna e pendente.

## Perimetro per task

| Task | Consegnato | Prova |
|---|---|---|
| F9.1 | BLAS in GpuMemory, scratch disgiunto, compaction differita, refit/rebuild e ritiro versionato | CPU lifecycle, F9-K1, deformazioni realmente eseguite, frame in volo, cambio scena e zero leak |
| F9.2 | Descriptor GPU per slot, mask/generazione, TLAS per frame ring, rebuild strutturale e refit | Corruzioni rifiutate, delete/reuse, mirrored, 1/2/3 frame in volo, 100K istanze con moto GPU |
| F9.3 | Proxy solo-indici con manifest/errore, Full per protezione/fallback, promozione runtime | S4/manifest Sponza; otto transizioni reali, compresi reassignment e full upload |
| F9.4 | Closest/any-hit, W&B, oracle CPU, probe, V-buffer e report | 36 casi funzionali, API/shader validation, negativi, tempi per raggio ×3 |
| F9.5 | Alpha generica, bypass opachi, IFT per PSO/slot, archive-assisted Compiler | Sponza MASK, reload positivo e poison su tutti i tre slot, anche con archivio attivo |

Topologia nuova passa per reload/cambio scena; il backend non dichiara
streaming incrementale di indici. Proxy e cono alpha mantengono il perimetro
di qualità dichiarato, senza promessa di equivalenza su ogni scena/LOD.
`--rt off` resta default; F10 aggiunge il primo consumatore di illuminazione.

## Verifica

- Debug e Release: **509 test, 1.774.699 assertion**, zero fallimenti/skip.
  Linux e Metal host syntax nella CI; snapshot Linux locale precedente ai
  cinque nuovi test di timing: 504 test /1.775.570 assertion.
- `tools/f9_check.py`: **36 funzionali +12 lifecycle +8 transizioni +4
  deformazioni**. Native e Apple9 forzato sul medesimo M5. Controlli negativi
  richiedono raw exit e marker corretti, mai un crash.
- `F9-K1` native/Apple9 sotto API+shader validation; 100K istanze con nuovo
  timing AS e checker; RT con timing disabilitato conserva tutti i raggi.
- Visual default/RT on: **zero pixel** diversi su otto scene.
- F6 quick, F7/F8 funzionale/qualità/contesto, lifetime MetalFX, archive
  (anche artefatto CI da macOS 26), hitch e hot reload: passati.
- Varianti mesh: 284 compatibili, peggiore 35 pixel/delta 1, 52 negativi,
  **PASS**. Indexed: gli stessi **28 fallimenti noti OPT-2.0** su Sponza
  debug 1/2 (48.081/69.903 pixel, delta 49/52), senza nuovi scostamenti.
- `leaks --atExit` con RT, trace e switch: **0 leak** usando il Compiler
  pubblico con archive lookup hints.
- V-buffer 640×360×32: Torus 513 /7.372.800 mismatch; Sponza 288 /7.372.800.
  Zero mismatch opaque interni in tutte le run; bordi/coverage/pareggi
  dichiarati, tutti i pixel inclusi. Alpha non viene promesso identico al
  raster a ogni LOD: il cono RT e le derivate raster hanno filtraggio diverso.

## Prestazioni e decisione

Contributo GPU di descriptor+TLAS, 100K istanze: **0,2213 ms native /
0,2189 ms Apple9 forzato**, mediana di tre medie, sotto 0,5 ms.
Native p95 delle repliche 0,257–0,340 ms, p99 fino a 0,784 ms e picco 2,153 ms:
non è una deadline garantita sul desktop. Default rebuildEvery **0**;
64/256 non danno un beneficio materiale e aumentano il p99.

Sponza 1080p, ns/raggio native: primary 0,2923, shadow 0,3268, AO 0,2220,
diffuse 0,5868. Secondari: tempo del probe finale separato dal setup primario.
A/B/A lungo contro `main` `2ccdf6b`: frame mediano −0,16% forward,
+1,22% temporal, −0,05% instances; p99 mediano entro 2,54%.
Zero allocazioni GPU nei preset stabili e zero compilazioni sul render thread.
Conservate tutte le repliche, incluse interferenze e prima serie breve rumorosa.

Dettagli, condizioni e serie: [perf-log](perf-log.md),
[risultati](results/F9-M5Max-2026-10-07.json),
[contratti/comandi](RENDERING_F9.md). Nessuna misura energetica o DRAM hardware.

## Difetti corretti durante l'integrazione

Near plane dei raggi; culling mirrored distinto dal facing mondiale;
baricentriche verificate con residuo spaziale FP32 limitato; emissive di
fallback non più protette inutilmente; trigger deformazione prima inattivo;
query di compaction obsolete anche prima del cambio versione; contatori
persi al reload; probe effettivo nel report; lifetime della funzione alpha
caricata direttamente dall'Archive; timestamp AS che anticipava il lavoro.

Il timestamp AS ora usa un join AS→Dispatch e un anchor prima del campione
preciso. La prima serie che riportava TLAS 0,3 µs è **invalidata**, conservata
e sostituita dalla serie v2. Il controllo rebuild-per-frame dà 1,585 ms,
distinguendo chiaramente il lavoro dal refit. Nessuna modifica a macOS,
installer o al contratto MetalFX temporale già risolto.

## Passo successivo e limite del proprietario

Integrare e verificare i commit F10–F12 consegnati dalla chat **Sviluppo
Phosphor**, poi F13–F14. La chat scrive soltanto codice/test/runner; questa
chat aggregatrice esegue le prove e decide l'accettazione. **Fermarsi dopo
F14 sviluppata e integrata**, sostituendo il precedente limite F27.
F14.5 e OPT restano candidati selezionabili, senza attivazione automatica.
