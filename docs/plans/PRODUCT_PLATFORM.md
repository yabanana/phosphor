# Piattaforma Phosphor: editor, Bevy, contenuti, 2D e UI

Decisione del proprietario, 2026-10-01. Il prodotto finale comprende un
renderer Metal specializzato **e** una piattaforma di sviluppo completa:
ECS/runtime, editor, contenuti della comunità, 2D, UI di gioco ed ecosistema.
Queste sono capacità funzionali richieste, non opportunità di ottimizzazione.
La [sequenza corrente](SEQUENCING.md) parte da F5 integrata e prosegue F6–F8;
i piani sono dettagliati in anticipo, ma i nuovi lavori si implementano
progressivamente senza migrare il runtime durante il consolidamento renderer.

## Direzione tecnica

**Prima opzione da verificare:** riusare Bevy ECS e, dove necessario per i
plugin, `bevy_app`, reflection, asset, scene, input e altri moduli Rust;
mantenere il renderer Phosphor C++/Metal con un confine di estrazione dati.
Bevy documenta l'uso standalone dell'ECS [R112] e il contratto dei plugin
sull'`App` [R113]. È una direzione di progetto da validare, non un'integrazione
già realizzata né la garanzia che basti disabilitare il renderer predefinito.

Il confine proposto trasporta snapshot/delta e comandi in batch con handle
versionati. L'autorità sul mondo di gioco resta unica; la GPU scene F5 è una
rappresentazione estratta. Attraverso Rust/C++ vanno definiti ABI, ownership,
thread, allocazione/rilascio, errori, lifetime e shutdown. Non attraversano
quel confine riferimenti ECS Rust o puntatori temporanei privi di contratto.
Event loop, clock, input e pool di worker hanno un proprietario esplicito.

**Alternativa:** ECS nativo C++ ispirato a semantica e API di Bevy, eventualmente
su EnTT/flecs. Evita di riscrivere storage e scheduling senza motivo, ma non
fornisce automaticamente compatibilità sorgente con plugin Rust. Prima di
sceglierla, confrontare i due percorsi con gli stessi casi; dichiarare il costo
di adapter/port e la copertura persa. Non chiamare «compatibile Bevy» una
somiglianza di architettura. La scelta si registra in F27.1, prima di costruire
editor e SDK su contratti definitivi.

## Compatibilità dell'ecosistema

Riferimento iniziale consultato: documentazione Bevy **0.19.1**; al kickoff
fissare una versione/revisione e le feature Cargo per l'intera suite. «Tutto
l'ecosistema» è l'obiettivo di riuso più ampio possibile, da verificare per
versione e piattaforma: il catalogo è aperto e non si può certificare in blocco.

| Classe | Strategia prevista | Prova necessaria |
|---|---|---|
| Logica ECS e plugin App senza dipendenze grafiche | Esecuzione nel vero host Bevy, ricompilato per la versione scelta | Lifecycle, eventi/messaggi, schedule e comportamento corretti |
| Asset, scene, input, animazione e fisica | Riuso dei moduli richiesti; adapter dei servizi Phosphor | Caricamento, aggiornamento, identità, timing e readback end-to-end |
| UI, inspector, picking, gizmo e 2D | Riuso dei dati/layout/logica quando separabili; estrazione e rendering Metal | Interazione e immagine equivalenti sui casi selezionati |
| Plugin dipendenti da `bevy_render`, wgpu, materiali o pipeline specifiche | Adapter o port mirato, se praticabile | Comportamento reale nel renderer Phosphor; nessuna compatibilità implicita |
| Plugin per altre piattaforme o feature non disponibili | Sostituzione, conversione o gap esplicito | Requisito equivalente verificato oppure stato non supportato |

Esempi da censire, **non dichiarati funzionanti**: `bevy-inspector-egui`,
`bevy_egui`, loader di asset e scene, input mapping, fisica 2D/3D, tilemap e
`bevy_ecs_ldtk`. L'inspector usa reflection e l'integrazione egui ha un
percorso di rendering Bevy [R119–R120]; LDtk può coinvolgere tilemap [R121].
Un importer riutilizzabile non dimostra che il relativo renderer sia riusabile.

F41 mantiene per ogni voce: versione, dipendenze, piattaforma, percorso
diretto/adapter/port, test, risultato e gap. «Supportato» richiede esecuzione;
compilazione o presenza nel catalogo non bastano. Plugin Rust si integrano
inizialmente da sorgente con dipendenze fissate, senza promettere ABI binaria
stabile o caricamento dinamico universale.

## Contenuti e formati della comunità

F21 fornisce un'interfaccia di importer/converter estendibile e un catalogo
di compatibilità. Per ogni formato documentare import diretto, conversione
tramite tool esistente, subset e limiti; materiali/rig/animazioni non si
considerano preservati per il solo caricamento della mesh. I loader Bevy
possono essere riusati quando i tipi e servizi richiesti sono disponibili
[R116], evitando parser duplicati.

La copertura di prodotto deve comprendere queste famiglie:

- **3D/DCC:** glTF/GLB, OpenUSD e percorsi da Blender; FBX/OBJ e altri formati
  diffusi tramite importer o conversioni esistenti. `.blend` è un workflow di
  esportazione/conversione da verificare, non una lettura universale presunta.
- **Immagini e materiali:** raster comuni, HDR/EXR, container KTX2/BasisU/DDS
  pertinenti; compressione runtime per Apple, spazi colore, mip, canali e
  conversione dei materiali con perdita esplicitata.
- **2D:** atlanti/spritesheet, animazioni Aseprite, mappe Tiled/LDtk, font
  TTF/OTF e contenuti vettoriali tramite percorso definito.
- **Audio/animazione/scene:** codec audio d'uso comune, scheletri e clip,
  asset Bevy/scene per versione e tipi registrati. Un'estensione di file
  supportata non implica tutte le sue combinazioni di feature.

Il catalogo viene ampliato con fixture provenienti dai workflow delle comunità,
provenienza e attribuzioni. Ogni rilascio dichiara la copertura e i gap; nessun
fallback silenzioso che perda texture, morph, unità, coordinate o animazioni.

## Parità funzionale 2D e UI

F39 e F40 assumono come riferimento una release Bevy fissata e i suoi esempi
[R114–R115, R118]. La parità è una matrice di comportamenti con prove, non
una lista di nomi di componenti. Eventuali capacità aggiuntive richieste da
un gioco completo restano distinguibili dalla parità con quella release.

**2D:** camere e viewport, sprite/atlanti/animazione, ordinamento trasparente,
mesh/materiali 2D, testo nel mondo, picking, tilemap tramite integrazioni,
pixel snapping e scaling, render-to-texture e composizione 2D/3D. Le primitive
si codificano nel render graph Metal condiviso; non si avvia un secondo
renderer solo per soddisfare un plugin.

**UI di gioco:** layout reattivo, flex/grid, unità/scaling, stili, testo e
font fallback, shaping, selezione/editing, clipboard e IME, focus, tab/controller,
mouse/touch, scrolling/clipping, immagini/atlanti, widget e stato, accessibilità,
localizzazione/RTL, transizioni e binding. Verificare ogni comportamento;
alcuni vanno oltre il supporto base della release Bevy consultata, ad esempio
la completezza dell'editing testo. La UI deve vivere nella build del gioco,
indipendentemente dall'overlay debug o dalla presenza dell'editor.

Riusare le librerie di layout, testo e accessibilità già adottate da Bevy quando
praticabile. Il layout e il renderer UI sono moduli distinti [R114/R117], ma
vanno censite le dipendenze transitive: nessuna separabilità totale presunta.

## Editor e workflow di prodotto

F34 è un editor di produzione per scene 2D/3D e UI: progetti, viewport,
outliner, inspector riflessivo, undo/redo, prefab/scene riusabili, asset browser,
materiali, animazione/VFX, play/pause/step, hot reload, salvataggio/recovery e
build. Il codice riusabile degli editor/inspector Bevy e della comunità viene
valutato prima di crearne uno equivalente. La shell può restare ImGui finché
utile, ma non sostituisce gli strumenti di authoring né la UI distribuita.

La prova di prodotto è un piccolo gioco 2D con menu/HUD/input controller e
una scena 3D editabile, costruiti con asset della comunità, plugin del corpus
e packaging Phosphor. F35 verifica regressioni e compatibilità; F38 consegna
SDK, esempi, template e documentazione della copertura. Una demo parziale
resta un milestone, non la dichiarazione di piattaforma completa.

## Sequenza e limiti attuali

1. Completare F5–F8 senza modificare in corsa ECS e renderer.
2. Prima degli editor/runtime estesi, F27.1/F27.7 provano l'host Bevy e il
   confine Rust/C++ su un caso reale; decidere riuso/alternativa con evidenza.
3. Attivare il nucleo F41 (corpus e contratto plugin) e F21 (importer), poi
   F39 e F40, e integrarli progressivamente nell'editor F34. La linea luce
   F9–F13 prosegue sui contratti del renderer esistente.
4. Chiudere le matrici funzionali/versioni prima di dichiarare completo il
   prodotto; ottimizzare soltanto i colli di bottiglia emersi.

**Vulkan:** unica ipotesi di lungo periodo, fuori dal lavoro attivo. Nessun
backend, port del legacy o RHI generale viene avviato per anticiparla. Il
renderer corrente resta Metal; la compatibilità dei plugin resta un tema
distinto dalla futura disponibilità di una seconda API grafica.
