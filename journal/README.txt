LOCUS journal manuscript (IRO Journals LaTeX template), compiled with pdfLaTeX.

  locus_journal.tex                   main article, with authors (camera-ready)      -> 25 pages
  locus_journal_anonymous.tex         main article, double-blind copy
  locus_supplementary.tex             supplementary material, with authors
  locus_supplementary_anonymous.tex   supplementary material, double-blind copy

One-command build: ./build.sh  (or compile each file twice; on Overleaf just compile the file you want).
j_references.tex holds the APA references (regenerate with make_refs.py); j_xref_main.tex / j_xref_supp.tex hold
the cross-document label numbers (regenerate with make_xref.py after editing either document).
Author photos: bios/dubey.jpg, bios/saxena.jpg.
