#!/bin/bash
# Full build of the main article and the supplementary material (named and anonymous).
cd "$(dirname "$0")"
run(){ pdflatex -interaction=nonstopmode "$1.tex" > /dev/null 2>&1; }
run locus_journal; run locus_supplementary
python3 make_refs.py; python3 make_xref.py
for m in locus_journal locus_supplementary locus_journal_anonymous locus_supplementary_anonymous; do run $m; run $m; done
for m in locus_journal locus_supplementary locus_journal_anonymous locus_supplementary_anonymous; do
  echo "$m: $(pdfinfo $m.pdf | grep Pages | tr -s ' ') | undefined=$(grep -c 'undefined' $m.log) | ??=$(pdftotext $m.pdf - | grep -c '??') | overfull=$(grep -c 'Overfull' $m.log)"
done
