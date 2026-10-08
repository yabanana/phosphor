# F12 — ReSTIR denso e proposta condizionata di preset full GI

## Nuovo confronto congelato

Il singolo caso `restir-dense-recovery-protocol-v1.json` (fd5032e) è stato
eseguito dal root e confrontato senza cambiamenti a immagini, soglie o algoritmo:
ReSTIR/custom16×8×16,64 raggi/probe,512frame, capture ogni frame, seed1001,
Cornell128×72, Le12→6 alframe256. Snapshot256 identico al post-step validato;
512checker runtime PASS. Il baseline fisico224..255 passa tutte le sei ROI.

Il gate congelato entro128frame **PASS**: primo PASS a+68, poi ricadute;
PASS continuo da+91 fino a128. A128 maxNRMSE29.664%, max|bias|14.368%,
residuo8.388% (trailing8.986%). Il precedente ReSTIR8×4×8 **resta FAIL**;
non viene sostituito nel ledger. Il margine sul limiteNRMSE30% resta piccolo.

|Offset|Max NRMSE|Max bias assoluto|Max residuo|Gate complessivo|
|---:|---:|---:|---:|---|
|0|86.29%|85.47%|84.93%|FAIL|
|1|68.50%|68.77%|66.59%|FAIL|
|4|53.77%|52.29%|48.96%|FAIL|
|8|46.22%|44.20%|40.30%|FAIL|
|16|41.08%|37.46%|33.09%|FAIL|
|32|33.96%|27.30%|25.22%|FAIL|
|64|30.26%|14.54%|10.52%|FAIL|
|128|29.66%|14.37%|8.39%|PASS|

Il gate include sia istantaneo sia trailing8. La convergenza F12≤128 resta
distinta dal rigetto della history F13≤4: il primo risultato non certifica
il secondo.

## Stabilità della coda: limite ancora visibile

È stata esaminata anche la coda disponibile129..255, mantenendo separata questa
diagnosi dall'esito preregistrato≤128. Non è corretto estendere il PASS a128
a una stabilità generale: cache e ReSTIR densi violano a tratti gli stessi
limiti sul frame istantaneo o trailing8.

|Preset custom|Frame con violazioni su127|Peggiore NRMSE istantanea|Ultima finestra32frame NRMSE|
|---|---:|---:|---:|
|DDGI16×8×16|**0**|23.94%|23.82%|
|Cache16×8×16|17|31.14%|29.57%|
|ReSTIR8×4×8|97|32.79%|30.97%|
|ReSTIR16×8×16|15|31.25%|29.58%|

ReSTIR denso viola solo il criterio NRMSE in questa coda, non bias/residuo;
cache densa viola anche il residuo (massimo11.72%). La media32frame finale
non cancella i fallimenti istantanei. Non è stata aumentata la finestra di
accettazione per far passare gli output. Questi risultati restano limiti di
qualità/variabilità dei preset, non un nuovo successo stabile dichiarato.

## Proposta per il prossimo confronto di costo

Candidato **full GI su M5 nativo**:

```
--render-path visibility --rt on --lighting restir
--gi ddgi --gi-grid 16x8x16 --gi-rays 64 --lighting-denoise custom
```

Motivazione: fra i preset verificati offre margine maggiore sul criterio
peggiore, passa la coppia thin-wall con negativo fisico rilevato, recupera
stabilmente da+87 a128 ed è l'unico senza violazioni nella coda129..255.
Il volume/scheduler/algoritmo restano quelli esistenti; nessun nuovo sistema
adattivo o rescaling radiometrico è stato aggiunto.

Cache/custom e ReSTIR/custom16×8×16 restano selezioni sperimentali con i limiti
qui dichiarati. Non diventano default solo perché più sofisticati. Un'adozione
richiede un vantaggio misurato sul corpus e una decisione esplicita sulla
variabilità osservata; il PASS limitato a128 non chiude quei residui.

**L'adozione del candidato DDGI è subordinata ai costi**, non fatta da questo
report. I precedenti numeri con validazione erano del frame completo e da un
solo run: non sono confronti prestazionali sufficienti. Il prossimo confronto
può mantenere questi preset già fissati, macchina quieta e tre repliche senza
API/shader validation, checker, capture o export; includere warmup e riportare
frameGPU p50/p95/p99, distribuzione CPU, allocazioni misurate, memoria engine e
device separatamente. Il baseline `--gi off` deve conservare gli altri segnali
per isolare il costo incrementale. Usare una scena/resoluzione rappresentativa
oltre alla Cornell128×72 di qualità; riportare l'esatto workload. Le statistiche
per-pass attuali con un campione restano diagnostiche, senza somme di mediane.

Solo se tali misure soddisfano il budget del prodotto, il full preset può essere
aggiornato e documentato con densità/raggi effettivi. Il tier ridotto/Apple9
forzato e altri dispositivi non ereditano queste misure o questa qualità.
Nessuna fase F12/F13 viene marcata completa da questa proposta.

## Evidenza

`docs/results/F12-dense-recovery-tail-M5Max-2026-10-07.json` contiene i129frame
congelati, provenance runtime e tutti i127frame della diagnosi di coda dei
quattro preset. Raw nel worktreef9-proxy: `build/f12-recovery-dense-v1/`.
Il root ha eseguito GPU e build; questo confronto è interamente CPU.
