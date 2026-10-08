# Piani dettagliati F e OPT

Aggiornato il **2026-10-03**, al checkpoint F7/F8 ([consegna](../F7_F8_HANDOFF.md), PR #14).

**58 piani, 414 task coperti.** Ogni piano contiene architettura, prerequisiti,
file esistenti/proposti, ordine dei pacchetti, spike, implementazione e
accettazione per singolo ID, ripiego ed evidenze da conservare. F6 contiene
inoltre il progetto operativo di dati, grafo, capacità, spike e runner.

La richiesta di dettaglio anticipato sostituisce le precedenti bozze brevi.
Non attiva il catalogo di ricerca e non inventa risultati sperimentali: le
scelte numeriche si fissano con la regola dello spike già scritta nel piano.
I file futuri sono destinazioni proposte; il kickoff riconcilia codice e SDK.

- **Checkpoint:** F5/F6 integrate; F7/F8 DEVELOPMENT_ACCEPTED M5; revisione post-F8 (2026-10-04): tempi per pass corretti, baseline M5 OPT-4.16 registrata, OPT-2 spostata dopo F13 con prerequisito OPT-2.0. F9 è integrata; la baseline F10–F14 e i limiti di qualificazione sono nel [handoff](../F10_F14_HANDOFF.md). [Revisione](../research/2026-10-04-post-f8-review.md).
- **Stop del proprietario:** F14. Nessuna F15/OPT viene attivata automaticamente; i criteri ancora aperti restano espliciti.
- **Piattaforma richiesta:** ECS/runtime, contenuti, audio, editor, 2D, UI e
  riuso Bevy, con implementazione progressiva e compatibilità verificata.
- **OPT/EDGE:** candidati selezionati per beneficio misurato, non passaggi obbligatori.
- **Audit:** F0–F8 e OPT-0/1 mantengono evidenze e residui storici; F7.4/F7.5 sperimentate e non adottate.

L’unico dispositivo disponibile è **M5 Max 128 GB**. Accettazione di sviluppo
qui, fallback pertinenti esercitati; certificazione fisica T0/altri chip
pendente senza bloccare la fase successiva. `--force-family apple9` è implementato ed esercitato: limita le capacità
e non emula il dispositivo fisico.

## Regole comuni

- [Priorità e attivazione](SEQUENCING.md).
- [Metodo, comandi e matrice delle verifiche](METHOD.md).
- [Decisione hardware, O12 e registro delle verifiche esterne](HARDWARE_VALIDATION.md).
- [Prodotto completo e riuso Bevy](PRODUCT_PLATFORM.md).
- [Dipendenze e metadati di pianificazione](dependencies.json).
- [Ricerca SoC H1–H12](../research/2026-10-01-apple-soc.md) e [bibliografia](../RESEARCH_REFERENCES.md).

## Catalogo

La colonna attivazione è una priorità di lavoro, non una dichiarazione di
implementazione. Tutti i documenti sono dettagliati; caselle e stato dei task
restano soltanto nella [roadmap](../ROADMAP.md).

| Piano | Fase | Attivazione | Task |
|---|---|---|---:|
| [F0](F0.md) | Fondamenta Metal 4 (manca solo la misura F0.8 su un dispositivo T0) | audit | 8 |
| [F1](F1.md) | Memoria, heap e residency | audit | 6 |
| [F2](F2.md) | Render graph e sincronizzazione automatica | audit | 8 |
| [F3](F3.md) | Pipeline: compilazione asincrona e archivi AOT | audit | 6 |
| [F4](F4.md) | Osservabilità | audit | 7 |
| [OPT-0](OPT-0.md) | Caratterizzazione del SoC  (manca OPT-0.2 su un T0) | audit | 6 |
| [OPT-1](OPT-1.md) | Memoria, grafo e banda come problema di ottimizzazione  (OPT-1.5 e OPT-1.7 parziali; T0 non misurato) | audit | 10 |
| [F5](F5.md) | GPU scene persistente e submission guidata dalla GPU | audit | 6 |
| [F6](F6.md) | Mesh shader e culling a due fasi | audit | 7 |
| [F7](F7.md) | Visibility buffer e shading ibrido TBDR | audit | 7 |
| [F8](F8.md) | HDR, EDR, esposizione e MetalFX temporal | audit | 7 |
| [OPT-2](OPT-2.md) | Shader, pipeline e occupancy | opportunita | 11 |
| [OPT-3](OPT-3.md) | Geometria, culling e dati di vertice | opportunita | 11 |
| [OPT-4](OPT-4.md) | Shading, banda e ricostruzione | opportunita | 17 |
| [F9](F9.md) | Infrastruttura ray tracing | preparazione | 6 |
| [F10](F10.md) | Ombre ibride | qualificazione parziale | 5 |
| [F11](F11.md) | Migliaia di luci | qualificazione parziale | 5 |
| [F12](F12.md) | Illuminazione globale | qualificazione parziale | 5 |
| [F13](F13.md) | Riflessioni, AO e denoiser | qualificazione parziale | 6 |
| [F14](F14.md) | Cielo, atmosfera, nuvole, meteo | qualificazione parziale | 5 |
| [OPT-5](OPT-5.md) | Budget di raggi e campionamento | opportunita | 11 |
| [OPT-6](OPT-6.md) | Illuminazione globale, cache e ammortamento | opportunita | 12 |
| [F15](F15.md) | Materiali avanzati e varianti shader | pianificato | 4 |
| [F16](F16.md) | Trasparenze on-tile, particelle, VFX | pianificato | 5 |
| [F17](F17.md) | Post-processing cinematografico | pianificato | 5 |
| [F18](F18.md) | Geometria virtualizzata | pianificato | 6 |
| [F19](F19.md) | Terreno, vegetazione, mondo procedurale GPU | pianificato | 5 |
| [F20](F20.md) | Acqua, oceano e fluidi | opportunita | 4 |
| [F21](F21.md) | Asset pipeline e cooker | requisito_pianificato | 9 |
| [F22](F22.md) | Streaming | pianificato | 6 |
| [OPT-7](OPT-7.md) | Geometria virtualizzata e mondo | opportunita | 10 |
| [OPT-8](OPT-8.md) | Streaming, texture e materiali | opportunita | 11 |
| [F23](F23.md) | CPU ultra-ottimizzata | pianificato | 9 |
| [F24](F24.md) | Fisica, cloth e distruzione | pianificato | 4 |
| [F25](F25.md) | Animazione e personaggi | pianificato | 5 |
| [F26](F26.md) | Audio di gioco e acustica avanzata | requisito_pianificato | 3 |
| [F27](F27.md) | Runtime di gioco ed ECS con riuso Bevy | requisito_pianificato | 10 |
| [OPT-9](OPT-9.md) | CPU, thread e memoria unificata | opportunita | 13 |
| [OPT-10](OPT-10.md) | Simulazione | opportunita | 8 |
| [F28](F28.md) | Scalabilità e presentazione | pianificato | 7 |
| [F29](F29.md) | Autotuning per dispositivo | opportunita | 5 |
| [F30](F30.md) | Rendering neurale I | opportunita | 7 |
| [F31](F31.md) | Rendering neurale II: reti addestrate in casa | opportunita | 5 |
| [F32](F32.md) | Path tracing in tempo reale | opportunita | 4 |
| [F33](F33.md) | Gaussian splatting ibrido | opportunita | 4 |
| [OPT-11](OPT-11.md) | Rendering neurale | opportunita | 11 |
| [OPT-12](OPT-12.md) | Autotuning, path tracing, splatting | opportunita | 10 |
| [OPT-13](OPT-13.md) | Scalabilità ed energia | opportunita | 8 |
| [F34](F34.md) | Editor e strumenti | requisito_pianificato | 8 |
| [F35](F35.md) | QA automatizzata | pianificato | 7 |
| [F36](F36.md) | Piattaforme Apple | pianificato | 3 |
| [F37](F37.md) | Vertical slice | pianificato | 4 |
| [F38](F38.md) | Distribuzione e live ops | pianificato | 5 |
| [OPT-14](OPT-14.md) | QA delle prestazioni e ottimizzazione continua | opportunita | 6 |
| [OPT-15](OPT-15.md) | Avvio, dimensioni e distribuzione | opportunita | 6 |
| [F39](F39.md) | 2D completo con riferimento Bevy | requisito_pianificato | 8 |
| [F40](F40.md) | UI di gioco completa con riferimento Bevy | requisito_pianificato | 10 |
| [F41](F41.md) | Ecosistema Bevy e SDK di estensione | requisito_pianificato | 7 |

## Consegna e manutenzione

Un pacchetto implementato produce commit, prove pertinenti e log; uno spike
produce anche scelta adotta/rimanda/scarta. Nessuna casella cambia per la sola
redazione del piano. Alla chiusura dichiarare l’ambito accettato sul M5, le
funzionalità ancora aperte e i dispositivi non certificati.

Aggiornare una specifica anticipata al kickoff o al cambio di un contratto.
Non occorre inseguire ogni dettaglio lontano a ogni commit. Restano vietati
risultati inventati, claim di supporto da semplice compilazione e nuove
infrastrutture di ottimizzazione senza carico reale e beneficio misurato.
