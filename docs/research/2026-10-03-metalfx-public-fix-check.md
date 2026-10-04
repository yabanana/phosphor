# F8.4 — nuova ricerca di una correzione senza isolamento

Verifica del 2026-10-03, dopo PR #15, base `cc38443`.

> **Aggiornamento 2026-10-04: superata.** La causa è stata trovata in-process
> (autoriferimento forte dello scaler nel suo filtro interno) e il motore la
> gestisce senza isolamento: [causa e rimedio](2026-10-04-metalfx-cycle-root-cause.md).
> Il confronto con macOS 27.0.1 stabile è **abbandonato**: il proprietario non
> vuole un volume di sistema aggiuntivo. Il volume `Phosphor Stable 27.0.1`
> (3,9 MB, solo copie del probe) è stato eliminato il 2026-10-04; l'installer
> da 17 GB e la cartella locale `build/metalfx-stable-27.0.1` citati sotto sono
> stati spostati nel Cestino. L'installazione non è mai partita.
**Esito: nessuna correzione pubblica applicabile verificata. F8.4 RIAPERTA.**
Il proprietario rifiuta il costo dei processi separati come soluzione finale.
Le misure precedenti restano valide per l'esperimento; l'accettazione della
fase viene ritirata. Nessuna prosecuzione OPT/F9+ finché questo gate non è
risolto. Nessuna modifica al percorso di rendering in questa indagine.

## Versioni effettive e ricerca

- Hardware: Apple M5 Max; macOS **27.2 beta 2, 26B5091g**; MetalFX **40.9**.
- Xcode selezionato: 27.0, `27A266a`. SDK diverso e runtime diverso restano
  due prove distinte: cambiare Xcode non sostituisce il framework del sistema.
- `softwareupdate --list`: **No new software available** sul canale corrente.
- `softwareupdate --list-full-installers` offre anche **macOS 27.0.1 stabile,
  26A434**, 17.955.378 KiB (circa 17,1 GiB di download). Non è stato scaricato
  o installato e non è stato eseguito alcun riavvio.
- L'[elenco Apple delle release](https://developer.apple.com/news/releases/)
  identifica la beta installata del 21 settembre e la stabile del 28 settembre.
  Le [note macOS 27.2](https://developer.apple.com/documentation/macos-release-notes/macos-27_2-release-notes)
  e [Xcode 27.2 beta 2](https://developer.apple.com/documentation/xcode-release-notes/xcode-27_2-release-notes)
  consultate non riportano una correzione di questo specifico ciclo.
- La guida d'integrazione di [Apple Game Porting Toolkit](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-metalfx-temporal-upscaler/references/integration-guide.md)
  non fornisce un'operazione pubblica di distruzione aggiuntiva. Anche gli
  header dell'SDK locale non espongono `invalidate`/`dispose` per lo scaler.
- Ricerca su Apple Developer Forums e issue/PR GitHub: nessuna correzione
  corrispondente verificata. La [PR mtld3d #454](https://github.com/athei/mtld3d/pull/454)
  corregge cache condivise e retirement per coda di **MTLFXSpatialScaler**:
  non corregge il ciclo interno del temporale. I vecchi fix MetalFX delle
  release 13/26 riguardano altri difetti, non provano la soluzione di questo.

L'assenza di una nota o di un risultato di ricerca non dimostra che Apple non
abbia una correzione interna. Non è stato inviato alcun feedback o messaggio
esterno. Non sono stati alterati canali beta, SDK selezionato o OS.

## Nuove prove sul M5

Ripetuta la matrice originale con **otto creazioni/rilasci** per percorso:

| Percorso | Scaler vivi dopo rilascio | Byte CPU classificati da leaks | Esito |
|---|---:|---:|---|
| Temporal Metal 4 | 8 | 2.349.376 | FAIL |
| Temporal Metal 3 | 8 | 2.203.360 | FAIL |
| Spatial Metal 4 | 0 | 0 | PASS, controllo |
| Denoised Metal 4 | 0 | 5.120 | FAIL |
| Denoised Metal 3 | 0 | 5.120 | FAIL |

Il temporal standard trattiene ancora un delta di allocazioni del device di
**167.133.184 byte (159,39 MiB)** a 640×360. Non sono byte persi per frame:
il trigger rimane la creazione/rilascio dell'istanza. I byte malloc e le
allocazioni del device sono domini diversi e non vanno sommati.

Esteso il [riproduttore pubblico](../../bench/f8_spike/metalfx_lifetime_matrix.mm)
per verificare nuove ipotesi senza modificare l'engine:

| Variante, otto istanze | Scaler vivi | Risultato |
|---|---:|---|
| Blit GPU completato prima della serie e dopo ogni rilascio, Metal 4 | 8 | FAIL; ciclo ancora visibile anche senza observer |
| Stessa prova, Metal 3 | 8 | FAIL |
| Upscale reale 640×360 → 1280×720 | 8 | FAIL |
| Upscale reale 640×360 → 1920×1080 | 8 | FAIL |
| RGBA8Unorm, Metal 4 e Metal 3 | 8 ciascuno | FAIL |
| RG11B10Float, Metal 4 | 8 | FAIL |
| Profondità R32Float, Metal 4 | 8 | FAIL |
| Asincrono, senza regione dinamica/reactive, autoexposure, 2×, blit | 8 | FAIL |

Il blit esegue una scrittura GPU reale e ne verifica stato e contenuto dopo
il completamento. È una prova di reclamation differita, suggerita da un
diverso caso di residency Metal; non un presunto rimedio ufficiale MetalFX.
Il controllo spatial con blit libera tutto; il denoised con blit continua a
perdere 640 byte CPU per creazione. Cambiare formato a RGBA8 è soltanto una
prova diagnostica, non una proposta di ridurre la precisione HDR del motore.

Le varianti non encodano lo scaler e non danno nuove evidenze di qualità.
Non si adotta una variante soltanto perché compila o perché `leaks` non
classifica un ciclo: servono distruzione weak **e** scansione senza observer.
Le prove reali di encoding, qualità e Swift ARC della precedente indagine
restano valide e separate.

Grezzi: `build/metalfx-public-fix-check/{baseline,followup}/`.
[Risultati compatti](../results/MetalFX-public-fix-check-M5Max-2026-10-03.json)
contengono hash della sorgente misurata, comandi, exit e campioni. Il binario
fallisce intenzionalmente il gate sul runtime difettoso: exit 1 non è PASS.

## Confronto decisivo ancora mancante

Non abbiamo eseguito la stessa prova su **macOS 27.0.1 stabile sullo stesso
M5**. Il difetto potrebbe essere una regressione della beta; è un'ipotesi,
non una diagnosi provata. Prima di costruire un altro workaround:

1. Avviare un'installazione separata di 27.0.1 stabile compatibile con questo
   Mac. Apple documenta l'[installazione su volume APFS aggiuntivo](https://support.apple.com/en-us/118282).
   Non sostituire l'OS attuale. L'installazione e il cambio di avvio richiedono
   autorizzazione esplicita: comportano modifica del disco e riavvio.
2. Eseguire il medesimo riproduttore, registrando build OS, MetalFX, hardware,
   hash binario/sorgente, weak liveness e scansione senza observer. I file
   preparati sono in `build/metalfx-public-fix-check/runtime-repro/`.
3. Se il temporale diretto libera tutte le istanze, eseguire i gate completi
   del renderer direct: ripetuti resize, DRS, camera cut, 1–4 viste, frame in
   volo, GPU validation, qualità temporale e prestazioni. Solo dopo scegliere
   la baseline OS e rimuovere il costo dei worker dal percorso adottato.
4. Se fallisce anche sulla stabile, il confronto estende il difetto a quel
   runtime. Preparare un feedback riproducibile per Apple e decidere una
   soluzione temporale alternativa interna al processo, con propri gate di
   qualità/DRS/memoria. Non dichiararla già equivalente a MetalFX o implementata.

Un test su un altro chip può aiutare, ma non prova la correzione sul M5.
La sola disponibilità di un installer, un SDK nuovo, un test in VM o una
build CI non sostituiscono la prova del runtime sul dispositivo reale.

La conclusione operativa resta **gate aperto**, non "SDK corretto" o
"isolamento accettato". Il nativo rimane utilizzabile; non equivale alla
ricostruzione temporale richiesta da F8.4.

## Prova stabile autorizzata — 2026-10-04

Il proprietario ha autorizzato la prova di macOS 27.0.1 stabile in un volume
separato, inclusi installazione e riavvio. È stato creato il volume APFS
`Phosphor Stable 27.0.1` nel contenitore del disco interno, senza sostituire
il gruppo del sistema attuale. Download dell'installer 27.0.1 completato con
`softwareupdate --fetch-full-installer`; il manifest nel payload conferma
**27.0.1 / 26A434**. App in `/Applications/Install macOS 27 Golden Gate.app`.
Il log del servizio Apple registra download e verifica del pacchetto dal CDN
Apple (prodotto `142-27367`). La verifica generica del bundle con codesign/spctl
segnala `obsolete (custom omit rules)`; il binario startosinstall supera codesign.
L'app Apple non modificata si apre normalmente; nessuna firma, quarantena o
protezione è stata alterata. Non si dichiara PASS del controllo generico fallito.

Stato operativo e identità del volume: `build/metalfx-stable-27.0.1/state.json`;
log del download: `build/metalfx-stable-27.0.1/fetch-installer.log`.
Il pacchetto autonomo è stato copiato in
`/Volumes/Phosphor Stable 27.0.1/Users/Shared/Phosphor-MetalFX-Probe`.
Il launcher `Run MetalFX Probe.command` non richiede Xcode/Python/rete,
registra build OS e hash del binario, verifica i due temporali con osservazione
weak e scansione senza observer, e richiede il controllo spatial positivo.
Ogni processo ha un limite di 90 secondi; la modalità normale rifiuta un OS
diverso da 27.0.1/26A434. Il controllo esplicito `--baseline` sulla beta ha
prodotto nuovamente `LIFETIME_FAIL` con spatial PASS: non è un risultato stabile.

Sul solo volume di prova è predisposto
`Library/LaunchAgents/org.phosphor.metalfx-stable-probe.plist`: avvia il test
una volta al primo accesso grafico alla build esatta. Non imposta riavvii e
non contiene credenziali. La registrazione temporanea usata per verificare
il guard nella sessione attuale è stata rimossa; sulla beta non resta un job
permanente. L'avvio automatico dopo l'installazione resta da verificare.

**Installazione, avvio della stabile e prova su quella build ancora pendenti.**
Installer fermo alla licenza prima di premere «Accetta»: richiesta la conferma
specifica prevista dal controllo UI per accettare il contratto. L'autorizzazione
all'installazione affiancata resta valida; non è stata richiesta una seconda volta.
La disponibilità del volume e del pacchetto non chiude F8.4. La normale
autenticazione del proprietario richiesta da macOS deve avvenire nella UI
del sistema; nessuna password va inserita nella chat o nei log.
