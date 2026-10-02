# Validazione con un solo dispositivo disponibile

Decisione del proprietario recepita il **2026-10-02**: il dispositivo disponibile
è **M5 Max, 128 GB**. Lo sviluppo e l'accettazione delle prossime fasi avvengono
qui. Non si aspetta un M3 e non si usa l'M5 come misura sostitutiva del M3.
Questa politica aggiorna O12 e l'interpretazione dei gate hardware di tutti
i piani F/OPT; non modifica retroattivamente le evidenze raccolte.

## Due esiti separati

| Esito | Prova richiesta | Effetto sul lavoro |
|---|---|---|
| `DEVELOPMENT_ACCEPTED` | Funzionalità dell'ambito scelto, correttezza, regressioni e budget sul M5 Max; fallback pertinenti esercitati sul medesimo dispositivo | Consente commit, integrazione e passaggio alla fase seguente |
| `HARDWARE_CERTIFIED` per modello/OS/preset | Esecuzione sul dispositivo fisico dichiarato, con qualità, prestazioni, memoria, termica e lifecycle pertinenti | Consente dichiarare supporto verificato soltanto per quella configurazione |
| `EXTERNAL_VALIDATION_PENDING` | Dispositivo o strumentazione non disponibili, con prova da eseguire e motivazione registrate | Non blocca sviluppo; blocca il claim di certificazione mancante |

Il M5 Max può coprire preset T2 e percorsi neurali T3 consentiti dalle sue
capacità: non prova ogni modello della stessa famiglia. T0 rimane il target
di compatibilità/prestazioni del prodotto futuro. Una release intermedia può
essere dichiarata verificata solo sul M5 Max; non può promettere prestazioni
T0 o supporto mobile certificato prima della prova fisica.

## Percorsi Apple9 sul M5

Al 2026-10-02 `soc_bench --force-family apple9` **esiste**;
`phosphor --force-family apple9` **non esiste ancora**. Il piano F6 include
l'aggiunta dell'override nell'app come parte dei contratti F6.1/F6.4/F6.7.
Fino a quell'implementazione non indicare la CLI dell'app come prova eseguita.

L'override deve:

1. Conservare device/famiglia fisici rilevati e limitare separatamente le
   capacità effettive usate per scegliere shader e API.
2. Disabilitare le specializzazioni Apple10 interessate; usare davvero il
   fallback, inclusi key di pipeline/grafo/piano e supporto dei formati.
3. Stampare nel report il percorso scelto. La sola accettazione del flag
   non dimostra che il fallback sia stato attraversato.
4. Non modificare nome chip, risultati del cost model, memoria fisica o
   etichetta della misura per far sembrare il dispositivo un M3.

Questo verifica **il percorso software compatibile**, sul driver/GPU M5.
Non emula il driver M3, throughput, cache, banda, consumi, limiti termici,
pressione di memoria o tutte le restrizioni di compilazione Apple9.
Affiancare audit delle API/feature table, compilazione dei target supportati
e test dei fallback. Restano necessari test fisici per la certificazione.

Un budget applicativo di 16 GB su 128 GB verifica eviction/backpressure e
comportamento ai limiti logici. Non riproduce il comportamento di un Mac con
16 GB, la contesa del sistema operativo o la sua banda.

## Budget e criteri di uscita

- Ogni nuova fase congela scena, risoluzione interna/output, qualità e
  workload prima della misura. I target T2 già espliciti restano tali.
- F6: Culling Viz sul M5 Max, 1920×1080 interno/output, scala 1, niente DRS
  o interpolation, seed e contenuto dichiarati; target 60 fps reali
  (p95 frame ≤16,67 ms, più CPU/GPU e p99 riportati). Misurare anche il
  fallback Apple9 sul M5. Se il target non passa, non dichiarare il gate
  locale superato: correggere o documentare una revisione motivata del preset.
- I target originari che nominano T0/T1/altri dispositivi restano obiettivi
  da certificare. Il preset corrispondente si prova sul M5 per correttezza;
  la sua prestazione sul M5 viene etichettata M5, senza fattori di scala.
- Nuove infrastrutture OPT mantengono il gate di beneficio sul frame
  disponibile: riferimento ≥3% o soluzione misurata di un limite concreto.
  Un hardware assente non giustifica il fallimento sul device disponibile.
- Sensori, contatori o misure input→fotoni non disponibili restano tali;
  proxy/stime possono aiutare a decidere ma non certificano il dato fisico.

## Caselle e registro delle verifiche esterne

Nessuna casella già completata viene alterata da questa decisione. Per task
nuovi che implementano una capacità e richiedono la prova T0 come verifica
trasversale O12, l'accettazione locale può chiudere il task; il report deve
registrare esplicitamente la certificazione esterna pendente. Per task il cui
oggetto è **la misura multi-device stessa**, la casella resta aperta finché
non è eseguita: non serve inventare una casella “mezza spuntata”.

| ID / ambito | Stato attuale | Come si chiude la porzione esterna |
|---|---|---|
| F0.8 | Misure storiche M5 disponibili; T0 pendente | Runner testbench su un T0 reale con manifest |
| OPT-0.2 | Suite M5 e percorsi Apple9 disponibili; T0 pendente | Suite SoC sul chip fisico, nuovo file risultati |
| F3.1, F5 e adozioni OPT-1: O12 | Evidenze M5 conservate; nessuna certificazione T0 aggiunta | Suite pertinente e confronto su T0 |
| F4.4 / OPT-1.5 | Contatori mancanti/parziali, indipendentemente da T0 | Accesso a contatori attribuibili; stime non chiudono il requisito |
| OPT-1.7 | Ridimensionamento dei compute reali da verificare | Misura sul carico reale: residuo tecnico, non automaticamente esentato dall'assenza T0 |
| F6.1/F6.4/F6.7 e uscita F6 | Da implementare; accettazione sul M5 native/fallback | Certificazione M3/Apple9 fisico aggiunta quando disponibile |
| F28.7 / F37.2 | Matrice multi-device ancora da implementare/verificare | Righe per dispositivi fisici; nessun tick integrale con il solo M5 |
| F36.1/F36.2/F36.3 | Port e hardware mobile/XR futuri | Compilazione più esecuzione sul target fisico; simulator non basta |
| OPT-13 / OPT-15.4 | Energia mobile / 8 GB non verificati | MacBook/dispositivo con configurazione pertinente e protocollo sostenuto |
| Ogni futura misura T0/T1/altro chip nei piani | `EXTERNAL_VALIDATION_PENDING` per default finché il device manca | Allegare risultati fisici prima del claim |

Il registro è estendibile nella consegna di ciascuna fase con ID, preset,
comando previsto e ragione della mancata prova. I residui tecnici sul M5
(bug, controlli falliti, funzionalità mancanti) restano bloccanti per l'ambito
che li richiede e non vanno riclassificati come verifiche esterne.

## Manifest e formula di consegna

Estendere il report quando si implementa l'override; i campi seguenti sono
il **contratto da implementare**, non una descrizione dello schema attuale:

```json
{
  "physical_device": "Apple M5 Max",
  "physical_family": "apple10",
  "memory_bytes": 137438953472,
  "effective_capabilities": "apple9",
  "preset": "phase-specific-frozen-preset",
  "validation_scope": "development",
  "unverified_devices": ["physical Apple9", "T0 base"]
}
```

Accompagnare con commit, OS/SDK, asset hash, seed, risoluzioni, qualità,
stato termico/alimentazione e dati grezzi del [metodo comune](METHOD.md).
Una consegna corretta dirà: “Ambito X accettato su M5 Max, con fallback
Apple9 esercitato; T0 fisico non disponibile, certificazione esterna pendente”.
Non dirà “M3 verificato” o “tutti i tier completati”.
