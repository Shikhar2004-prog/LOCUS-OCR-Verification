## What problem does this solve?

An OCR engine always returns a transcription, including when it is wrong. Most pipelines then pass that text straight
into accounting, claims processing, search or archival systems. A single misread digit ("640" for "6.40", "48,730" for
"48,750") becomes a wrong payment, claim or record. Today the choice is to check every word by hand (expensive) or
to trust the engine's own confidence. That confidence is often high on exactly these errors: Tesseract reported 0.92
for "640".

**LOCUS** adds a verification step after any OCR engine. For each word it returns:
* **P(correct)**, a calibrated probability that the word is right, so a team can choose how much human review to buy;
* **a repair proposal** for flagged words;
* **an explanation**: the rival strings it considered and how strongly the image supports each of them.

## How it works (one word)

1. **Counterfactual string graph.** Around the claim *h*, generate the strings the engine could have confused it with:
   confusable characters (0/O, 1/l, 5/S, 8/B ...), case flips, deletions and insertions, and the readings of two frozen
   recognisers (TrOCR-base and PARSeq).
2. **Render-normalised evidence.** For each hypothesis *v*:
   ρ(v) = s(v | real image) − s(v | clean rendering of v).
   Subtracting the score on a clean rendering removes the recogniser's bias towards common strings. What remains is how
   well *this* image supports *v*.
3. **GNN with a local-optimality readout.** A GATv2 network reads the graph and asks whether the claim is the best
   explanation among its neighbours (P_opt). It is pretrained without labels.
4. **Few-shot head.** Only **25 labelled words per class** adapt it to a new document type.

## Benchmark results (from the paper; k = 25 labels per class)

| | FUNSD | CORD | SROIE | IAM | IIIT5K | Avg |
|---|---|---|---|---|---|---|
| Engine confidence (F1 / AUROC) | .756 / .861 | .777 / .892 | .780 / .859 | .802 / .878 | .788 / .938 | .781 / .886 |
| PARSeq re-OCR, graded | **.946 / .976** | .973 / .986 | .742 / .760 | .879 / .914 | **.993 / .995** | .907 / .926 |
| Qwen2.5-VL-7B judge | .760 / .846 | .891 / .945 | .866 / .941 | .700 / .788 | .888 / .974 | .821 / .899 |
| **LOCUS** | .943 / .974 | **.978 / .993** | .973 / **.992** | **.906 / .969** | .971 / .991 | **.954 / .984** |

**Human-review workload** is the share of words a person must check so that the auto-accepted words reach 95%
precision. LOCUS needs 37% on FUNSD (Tesseract confidence: 66%), 37% on CORD (59%), 45% on SROIE (91%), 51% on IAM
(91%) and 26% on IIIT5K (37%).

## Honest limitations

* A strong second recogniser (PARSeq) is a hard baseline. It ties LOCUS on FUNSD and beats it on IIIT5K. On the
  synthetic demo pages bundled here, simply flagging disagreement with PARSeq catches about as many errors as LOCUS.
  The advantage of LOCUS is consistency: the best re-reader changes from dataset to dataset (PARSeq reaches only F1
  .742 on SROIE receipts, where TrOCR-base is strongest), while LOCUS is near the best on all five.
* Its exact repair proposals are good but not best everywhere. Substituting PARSeq's reading repairs more errors on
  four of five datasets.
* **Speed:** about 18 words/s on one RTX A6000, because it runs two frozen recognisers on the image and on every
  rendering. That is slower than PARSeq alone (372 words/s), but fine for back-office batches.
* Latin script, English benchmarks. Words that Tesseract segments wrongly (several words merged into one box) are
  flagged, but cannot be repaired word by word.
* The model shipped with this app is a single seed (seed 0) with the fusion head, exported by `app/export_models.py`.
  Its held-out test scores are shown in the app and match the paper's ensemble to within about 0.01.
