# Prosecuzione F9–F27

Autorizzazione del proprietario: 2026-10-07, proseguire lo sviluppo fino a F27.
Questo registro coordina i piani esistenti; non sostituisce le caselle della
[roadmap](../ROADMAP.md) né dichiara implementate le fasi future.

## Checkpoint

- F7/F8 e revisione post-F8 integrate (PR #16/#17); indagine OS abbandonata.
- F9 in corso su `phase/f9`, con [piano approvato](F9-EXECUTION.md), spike
  misurati e integrazione BLAS/TLAS/traversal/proxy/diagnostica in verifica.
- F10–F12 assegnate il 2026-10-07 alla chat **Sviluppo Phosphor** per sola
  scrittura in worktree separato da `7997f12`: sorgenti, shader, test e runner
  non eseguiti. Questa chat aggregatrice conserva F9, review, build, prove
  CPU/GPU, misure, integrazione e merge. La consegna di codice non costituisce
  accettazione della fase.
- OPT-2 resta dopo F13; i candidati OPT/EDGE richiedono un consumatore e un
  beneficio misurato. F9.6 non è attivato.

## Ordine per dipendenze

1. Chiudere F9: correttezza CPU/GPU, lifecycle, proxy misurati, V-buffer,
   regressioni, prestazioni, documenti, PR e CI prima del merge.
2. F10–F13: ombre, luci, GI, riflessi/AO/denoiser. F10.3 richiede F11.1;
   l'uscita denoised di F11 richiede F13, senza bloccare la baseline DI.
3. Materiali/post e mondo: F15–F17, F14 quando il corpus outdoor lo rende
   concreto. F14.5, F16.5 e l'intera F20 restano candidati.
4. Anticipare il nucleo F21.3 prima di F18; prima geometria virtualizzata
   residente, poi streaming F22. F19.2 dipende da F22.3: evitare il ciclo
   fra intere fasi F18/F21/F22.
5. Anticipare i contratti host/bridge F27.1/.7 prima di consolidare F23–F26,
   per mantenere un solo mondo, scheduler e sistema asset/input. F27 dipende
   già da F5/F8; F26.1 richiede il suo runtime minimo.
6. Completare i pacchetti selezionati di F21–F27: asset/community, streaming,
   servizi CPU, fisica, animazione, audio e host/ECS, verificando i consumatori
   integrati e conservando i limiti hardware espliciti.

Ogni kickoff riconcilia codice, API e dipendenze effettivamente disponibili.
Non si adottano tutte le ipotesi lontane per il solo fatto che esiste un piano.

## Primo contratto F10 da verificare dopo F9

- Raster depth/V-buffer → punto e normale geometrica → offset W&B orientato
  alla luce → raggi sul disco solare → shadow any-hit/alpha LOD0 → mask.
- Il material resolve modula soltanto la luce diretta interessata; non
  moltiplica HDR complessivo, emissive o illuminazione ambientale.
- Questa prima soluzione non richiede un nuovo guide pass prima della luce,
  evitando un ciclo guide → shadow → material resolve.
- Il consumer riceve risorse RT readonly per slot; la sua IFT appartiene al
  proprio PSO e segue hot reload/retirement, senza riusare la IFT diagnostica.
- Le CSM usano liste caster per cascata, indipendenti dal culling della camera.
- Storia per vista/segnale; controlli negativi su bias, caster, cache e history.

## Regole di consegna

La chat di sviluppo consegna commit distinti per fase e un handoff con
contratti, assunzioni, comandi di verifica e stato **NON VERIFICATO**. Non
esegue build, renderer o benchmark, non modifica il checkout dell'aggregatore
e non spunta la roadmap. L'aggregatore legge i risultati, testa e integra
in ordine di dipendenza; eventuali correzioni tornano al pacchetto interessato.

Una fase è accettata soltanto per il perimetro provato sul M5 Max. Famiglia
Apple9 forzata non è certificazione M3. Restano necessari immagini, risultati
GPU reali e controlli negativi, oltre a compilazione e test CPU. Le misure
prestazionali usano un solo workload GPU e una macchina quieta; i numeri di
correttezza non diventano implicitamente misure di performance.

Nessuna modifica a volumi, installer, avvio o sistema operativo. Nessun
messaggio/feedback esterno senza autorizzazione specifica del proprietario.
