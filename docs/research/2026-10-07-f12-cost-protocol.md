# F12 — protocollo di costo GI fissato prima delle misure

Pronto per il root, **nessun renderer o benchmark eseguito da questo agente**.
`tools/testdata/f12/cost-protocol-v1.json` fissa18run: Cornell e Sponza,
tre triplette A1/B/A2 per scena.512frame misurati dopo128warmup,1920×1080,
post nativo, finestra visibile, noVSync, GPUtiming seriale. UI disabilitata.
Gli altri segnali sono identici: RTon, DIReSTIR, shadows/reflections/AOoff,
custom denoise. A usaGIoff; B DDGI16×8×16 con64rays. Il tier è M5 nativo.

Sono mantenute le policy default dell'archive e della compilazione pipeline;
i loro contatori sono conservati. Niente validation, checker, capture/export,
Tracy, upscaler temporale, DRS, EDR o esposizione automatica. Il root mantiene
la finestra/display visibili e la macchina quieta, un processoGPU alla volta.

## Verifica dell'interfaccia e staging

Verificati nel codice reale: `--post` è un flag senza argomento;
`--upscaler native` abilita post; `--gpu-timing-serial` attende il completamento
GPU di ogni frame; `--resolution` controlla i pixel effettivi del drawable.
SceneViewer è bench4 e `--scene` esplicito impedisce il fallback procedurale.
Cornell fisica è bench6 con `--lighting-scene cornell`. I controlli del volume
sono vietati conGIoff, quindi A non riceve `--gi-grid`.
Il CMakeCache letto è Release, PHOSPHOR_BUILD_APP=ON e PHOSPHOR_TRACY=OFF;
il tool ricontrolla questi campi al momento dello staging finale.

Dopo l'ultimo build e il gate nativo, dalla checkout di integrazione:

```
python3 tools/f12_gi_cost.py stage \
  --protocol tools/testdata/f12/cost-protocol-v1.json \
  --project . --app build/lighting/phosphor \
  --sponza /Users/danielsan/Documents/phosphor/assets/sponza/Sponza.gltf \
  --output build/f12-full-gi-cost-v1
```

Il percorso app può indicare il build finale scelto dal root. `stage` non
avvia l'app: produce `plan.json` con18argv concrete, `checked_command` per
`run_checked.py`, working directory, ambiente da ripulire, hash del binario,
metallib/archive, glTF e risorse esterne, commit/diff e configurazione CMake.
Eseguire i `checked_command` nell'ordine dichiarato e dalla working directory
indicata. Non cambiare sorgenti, build o asset durante la batteria.

```
python3 tools/f12_gi_cost.py analyze \
  --plan build/f12-full-gi-cost-v1/plan.json \
  --output build/f12-full-gi-cost-v1/analysis.json
```

## Analisi e limiti

Sono obbligatori processo/EXIT0, stessa configurazione/provenienza, risoluzione
nativa effettiva, assenza dei checker, GPU failures0 e **O7: allocazioni GPU
misurate esattamente0 in ogni run**. Vengono respinti anche artefatti modificati,
run sovrapposti/riordinati e tempi GPU non positivi/non finiti.

Per ogni tripla, il baseline è la media degli statistici A1/A2; il delta è
B−baseline. Drift A1/A2 oltre5% sulp50 o10% sulp95 invalida l'inferenza temporale
di quella tripla. Sono limiti di validità sperimentale preregistrati, non un
budget del prodotto. Si conservano tutti i run, senza scegliere il più veloce,
scartare outlier o ritentare automaticamente. Le tre repliche valide producono
singoli delta, mediana, min/max e deviazione standard campionaria.

Le quantili non vengono poolate e le mediane dei pass non vengono sommate.
Ogni unità conserva il proprio numero di campioni; attribuzione steady richiede
almeno90% dei512frame per unità. Copertura incompleta dei pass non annulla la
distribuzione dell'intero frame, ma impedisce di presentarla come costo stabile
per ogni pass. Memoria engine, allocazioni device e footprint processo restano
misure separate, con delta accoppiati.

Nessun budget numerico di tempo/memoria è stato inventato. Il risultato sarà
evidenza per decidere il default, non una decisione automatica: nessuna misura
M3, prestazione complessiva F14 o chiusura di fase viene inferita.

Test leggero `python3 tools/test_f12_gi_cost.py`: report sintetici per18run,
delta atteso2ms, copertura pass insufficiente conservata; negativi O7,
drift e ordine/concorrenza correttamente respinti. Durata circa0.05s, nessuna
applicazione o GPU avviata. Il caso di test è esplicitamente sintetico e non
costituisce evidenza di prestazioni del motore.
