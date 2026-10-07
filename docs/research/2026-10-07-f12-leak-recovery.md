# F12 — leakage fisico e recupero dopo un cambio di emissione

## Esito congelato

La coppia thin-wall **PASS** rileva il negativo tramite la metrica di leak,
mentre entrambi i run superano i normali invarianti GPU. Il recupero entro
128 frame passa per DDGI/custom e cache/custom16×8×16; **ReSTIR/custom8×4×8
FAIL**, short_box NRMSE31.13% contro limite30%. Nessuna soglia è cambiata.
È stato preregistrato un solo caso aggiuntivo ReSTIR/custom16×8×16, ancora
non eseguito in questo report. Il precedente FAIL rimane un risultato separato.

## Thin-wall, DDGI/custom16×8×16

Stesso binario/configurazione fisica/snapshot e capture ID8; unica differenza
`--debug-gi-no-visibility` e relativo campo report false/true.512frame, media
32capture256..504,256×144, seed1001. Reference indipendente16384spp×2seed,
849pixel bright e830dark geometrici, già congelati.

| Metrica |Positivo|Visibilità disabilitata|Limite|
|---|---:|---:|---:|
|Bright NRMSE|6.47%|6.85%|30%|
|Bright bias|−3.81%|−4.51%|±20%|
|Errore positivo dark medio /meanBright|5.44%|7.51%|10%|
|Errore positivo dark p95 /meanBright|7.92%|**25.669%**|25%|
|Invarianti runtime|PASS|PASS|PASS|

Il negativo fallisce proprio il gate leakage, non un counter artificiale,
il bright-control o un NaN. Il margine sul p95 è0.669 punti percentuali;
la sensibilità ai due seed indipendenti dell'oracolo mantiene il FAIL:
25.682% e25.656%. Il positivo resta7.946%/7.892% nei due confronti.
La reference mediata resta quella ufficiale: nessun seed è stato selezionato.
Questo risultato non significa leak matematicamente zero e non estende la
certificazione alle modalità cache/ReSTIR thin-wall non eseguite qui.

## Recovery emissive-step

Snapshot256 dei tre run identico al post-step già validato: Le12→6, tutto il
resto fisicamente invariato. Si usa l'oracolo esatto0.5×Cornell, senza fitting.
Capture ogni frame; baseline224..255, poi ogni offset0..128. Il gate richiede
sia istantaneo sia trailing8 entro20%bias/30%NRMSE e residuo≤10%. I baseline
fisici passano tutti. Le immagini vecchia trattenuta e nera falliscono i controlli
negativi. Il campo "residuo" misura il vecchio stato rispetto alla risposta
stazionaria scalata, senza sostituire il confronto fisico con l'oracolo.

| Caso |Primo PASS|PASS continuo fino a128|Esito128|
|---|---:|---:|---|
|DDGI/custom16×8×16|87|87|PASS|
|Cache/custom16×8×16|60|124|PASS|
|ReSTIR/custom8×4×8|mai|mai|FAIL|

Le ricadute della cache vanno conservate: non è corretto presentare60 frame
come recupero stabile. "Continuo" si riferisce a tutti i frame osservati fino
al limite128, non a una garanzia indefinita dopo quel limite.

Massimo residuo per ROI istantaneo ai checkpoint (il report JSON contiene
anche tutte le NRMSE/bias e la media trailing8, per ciascuno dei129frame):

|Offset|DDGI dense|Cache dense|ReSTIR base|
|---:|---:|---:|---:|
|0|62.25%|85.22%|84.85%|
|1|61.30%|64.31%|66.19%|
|4|58.85%|54.40%|48.41%|
|8|55.75%|39.25%|39.45%|
|16|49.53%|26.04%|31.96%|
|32|33.70%|17.80%|25.41%|
|64|15.01%|8.57%|10.55%|
|128|5.93%|6.51%|9.37%|

A128 ReSTIR passa il residuo (trailing9.81%) e il bias massimo (short_box−19.33%),
ma fallisce NRMSE short_box31.13%. Non è dunque il solo gate temporale a
ritardare: resta qualità fisica insufficiente nel preset corrente.

## Diagnosi, senza nuovo algoritmo

La sorgente invalida atlas/cache/reservoir al cambio radiometrico
(`gi_passes.cpp`, signal/content epoch). `giIrradiance` restituisce zero per i
bounce precedenti sul frame di reset; il trace riparte da illuminazione diretta
alle superfici secondarie. Dal frame seguente, `ddgi_blend` usa hysteresis0.95
per ricostruire il trasporto con più rimbalzi. F13 collega la revisione della
history GI a cacheGeneration e resetta il contenuto per-signal.

L'inizio scuro (fino a circa−85%) è coerente con questo riavvio, insieme al
rumore iniziale e al filtro, non con la persistenza di una history ancora
luminosa aLe12. I frame224/232/240/248 prima dello step sono bit-identici ai
precedenti run statici in tutte e tre le modalità: il nuovo flag diagnostico
afalse e la frequenza dei capture non hanno cambiato quegli output.
Questi dati/source sono evidenza sul meccanismo; **non certificano il requisito
separato F13 di rigetto history entro4 frame**, che richiede i suoi controlli.

Un'analisi esplorativa della coda, senza cambiare l'esito≤128, trova per ReSTIR
nella media degli offset224..255: NRMSE short_box30.97%, bias−15.37%, massimo
residuo4.91%. Aspettare di più non elimina automaticamente il limite spaziale
del preset/approssimazione. La DDGI8×4×8 sottostante era già carente sul box corto;
16×8×16 aveva migliorato DDGI/cache. Il nuovo caso singolo è congelato in
`restir-dense-recovery-protocol-v1.json` (fd5032e), con stesso algoritmo, raggi,
hysteresis, ROI e soglie. ReSTIR dense non è ancora dichiarato PASS.

Un'eventuale gestione esatta di uno scaling radiometrico globale potrebbe
preservare/rescalare history compatibili, ma richiederebbe un contratto proprio
su cache, reservoir/PDF e invalidazioni. Non è stata implementata: prima viene
il semplice confronto di preset autorizzato.

## Evidenza e ambito

Dati completi: `docs/results/F12-leak-recovery-M5Max-2026-10-07.json`.
Raw nel worktreef9-proxy: `build/f12-thin-reference-v1/quality-v1/`,
`build/f12-recovery-v1/`. I JSON conservano provenance runtime e snapshot,
checkpoints intermedi, tutti i frame e negativi. Run GPU eseguiti dal root;
questo agente ha soltanto confrontato gli output con strumenti CPU.
Nessun benchmark prestazionale, altra scena, dispositivo fisico o intera fase
F12/F13 viene certificato da questi risultati locali.
