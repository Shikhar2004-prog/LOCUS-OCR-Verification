# LOCUS demo (no install)

Open `index.html` in any browser (Safari, Chrome, Firefox). You do not need a server, a GPU or Python. Keep `data.js` and `LOCUS_demo.mp4` in the same folder.

* `index.html` + `data.js`: the real outputs of the exported CORD-trained LOCUS model (k = 25, seed 0) on three synthetic
  demo pages: a degraded invoice, a faded receipt and an insurance claim form. That is 298 Tesseract words, 78 of them OCR errors.
  For every word the page stores:
  * the crop;
  * Tesseract's claim and confidence;
  * the true text;
  * P(correct) and the repair;
  * the TrOCR-base and PARSeq readings;
  * the counterfactual string graph (ρ under both recognisers, P_opt) and clean renderings R(v).

  The pages are synthetic, so the true text of every word is known. Nothing is hand-picked: every word on each page is included, together with the model's misses and false alarms.

  Tabs:
  * **Verify a document:** red and green boxes on the page. The review threshold is a live slider. LOCUS is compared with Tesseract's confidence at the same review budget and with PARSeq disagreement. The page also shows the flagged words and the text after review.
  * **Inspect a word:** click any word to see its verdict, graph and hypothesis table.
  * **How it works:** includes the video.
  * **Results:** the paper's benchmark tables and review workload.
* `LOCUS_demo.mp4`: the 2.5-minute video.

Rebuild `data.js` with `python app/make_static_demo.py`; this needs a GPU and the research environment.
To verify your own documents, use the full application in `app/` (`python app/app.py`).
