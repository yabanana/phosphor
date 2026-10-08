# F12 — fixture emissive-step e protocolli filtrati/leak/recovery

`emissive-step` riusa esattamente la Cornell statica: unica lampada fisica,
nessuna luce aggiuntiva, geometria/camera ferme. Il primo update con dt>0
precede il frame0; i primi256 update positivi mantengono Le12 (frame0..255),
il257esimo imposta Le6 (frame256), poi la scena resta ferma. Gli update dt0
non contano e non sporcano ECS. La modifica del materiale avviene una volta;
il reset setup riporta il contatore e Le allo stato iniziale.

La prova CPU standalone usa `tests/test_lighting_validation.cpp` e il nuovo
sorgente della fixture, linkati con una copia del core già compilato del root.
**13 test /35.835 assert PASS**. La regressione controlla385frame, dt0
intercalati, dt positivi diversi, unica material delta alframe256, nessuna
transform delta, nessuna luce nascosta, materiale GPU estratto ai frame255/256
e riarmo dopo teardown/setup. Non è un test GPU: root deve esportare e
confrontare gli snapshot reali255/256 per provare il confine nel loop Engine.

Il protocollo preregistrato è in
`tools/testdata/f12/leak-recovery-protocol-v1.json`. I checkpoint recovery sono
0,1,4,8,16,32,64,128frame dopo la transizione. La convergenza F12≤128frame
è distinta dal requisito F13 di rigetto della vecchia history≤4frame; il report
deve mostrare entrambi e non sostituire il secondo con il primo.

L'oracolo post è0.5×il riferimento Cornell già validato: il trasporto lineare
con un'unica emissione e materiali fissi lo impone. È consentito riusarlo solo
se gli snapshot dimostrano che l'unica differenza fisica è Le12→6, comprese le
distribuzioni degli emettitori. Frame/revisioni diagnostici possono cambiare;
geometria, texture, camera, sky e altre luci devono restare invariati.

Per non confondere la coda temporale con il bias stazionario dell'approssimazione,
il residuo usa la legge di scala anche sul candidato: per ogni ROI,
abs(mean(C_post)−0.5mean(C_pre))/(0.5mean(C_pre))≤10% entro+128.
C_pre è fissato ai frame224..255. Le stesse soglie fisiche20%bias/30%NRMSE
contro l'oracolo restano obbligatorie: non si può usare come baseline una
soluzione già fuori soglia. Vanno riportati tutti i checkpoint, il primo
PASS osservato e se resta PASS nei successivi; nessun semplice checkbox128.
Un'immagine vecchia trattenuta e una post completamente nera devono fallire.

Il confronto filtrato statico è congelato separatamente in
`tools/testdata/f12/filtered-protocol-v1.json`: cache/custom base e16×8×16,
DDGI/custom16×8×16, ReSTIR/custom base,512frame seed1001 capture8.
Si legge `indirect-diffuse-filtered` (ID8), il vero E selezionato dal custom
F13 moltiplicato una volta per albedo×(1−metallic)/pi. Non si filtra offline
una surrogate. Le sei ROI Cornell e le soglie20/30 restano identiche.
Il precedente FAIL rawcache rimane nei suoi report, anche se il filtrato passa.

Thin-walls usa la fixture esistente: separatore chiuso10mm, lampada a sinistra,
possibili rimbalzi fisici attorno all'apertura frontale. ROI bright/dark definite
solo da intersezioni geometriche e dal segmento verso il centro lampada,
erose2px, almeno16pixel per maschera a256×144. L'oracolo non assume che dark
sia zero. Errore positivo medio/p95 in dark limitato a10%/25% della luminanza
media reference bright; bright mantiene20/30. La scelta delle maschere deve
avvenire sul solo snapshot prima di aprire immagini candidate. Snapshot,
riferimenti e misure thin-wall/recovery restano da eseguire.

Strumenti CPU aggiunti:

- `tools/f12_fixture_regions.py`: consuma solo snapshot/PLY, individua per
  dimensioni il separatore10mm e la lampada, costruisce bright/dark senza leggere
  alcuna immagine candidata. In un test geometrico sintetico costruito dalle
  dimensioni della fixture produce849pixel bright e830dark dopo erosione sia dei
  label delle superfici sia delle maschere geometriche. Questa è verifica
  del programma, **non** coverage misurata sullo snapshot runtime ancora atteso.
- `tools/f12_step_snapshot_check.py`: pretende i frame255/256, asset byte-identici,
  stessa camera/geometria/material response e sola emissione dimezzata, poi
  deriva il PFM post dall'oracolo indipendente accettato. Testato su snapshot
  sintetici: accetta lo scaling esatto e rifiuta drift della camera prima di
  generare qualsiasi riferimento. Non dimostra ancora la transizione GPU.
- Il collector filtrato rifiuta capture raw al posto dell'ID8 e verifica la
  selezione `--lighting-denoise custom`; positive/negative metadata test PASS.

- `tools/f12_recovery_compare.py`: misura ogni frame0..128 dopo il cambio,
  riporta tutti i checkpoint e il primo PASS che rimane tale fino a128.
  Al checkpoint128 richiede sia l'immagine istantanea sia la media trailing8
  entro i gate fisici e residuo≤10%; la media non sostituisce il requisito
  istantaneo. Il baseline224..255 deve già passare i gate fisici.
  Test sintetico di cambio istantaneo esatto PASS aoffset0; negativi immagine
  vecchia e immagine nera FAIL. Sono unit check dello strumento, non prove
  di convergenza del renderer. Il report include gli hash dei protocolli.
