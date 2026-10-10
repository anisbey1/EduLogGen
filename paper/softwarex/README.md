# SoftwareX manuscript

| File | Content |
| ---- | ------- |
| `paper.tex` | Manuscript (elsarticle, SoftwareX OSP structure) |
| `references.bib` | References (DOIs checked against Crossref) |
| `fig_architecture.pdf` | Fig. 1, from `make_architecture.py` |
| `fig_fidelity.pdf`, `table_*.tex` | Fig. 2 and Tables 1–2, from `studies/oulad/report.py` |
| `cover_letter.md` | Cover letter |

## Build

Upload this folder to Overleaf (compiler: pdfLaTeX, main file `paper.tex`),
or run `latexmk -pdf paper.tex` locally.

To regenerate the results, figures, and tables:

```bash
python studies/oulad/prepare.py --oulad OULAD --output studies/oulad/data
python studies/oulad/run.py --data studies/oulad/data --oulad OULAD --output studies/oulad/out
python studies/oulad/report.py --results studies/oulad/out/results.json --output paper/softwarex
python paper/softwarex/make_architecture.py
```

## Submission checklist

- [ ] Compiles on Overleaf without errors; figures and tables placed sensibly
- [ ] Compare with the current SoftwareX OSP template from the Guide for
      Authors (code metadata wording, section order); adjust if it changed
- [ ] Word count ≤ 4,000 (currently about 2,450, including abstract,
      captions, and code)
- [ ] AI declaration describes your actual use of AI tools
- [ ] Funding statement is correct (currently: no specific grant)
- [ ] Code metadata C1/C2 match the release cited (v1.4.2); C3 Zenodo DOI
- [ ] Repository public, README with installation, MIT licence (done)
- [ ] Suggested reviewers (often requested by the submission system)
- [ ] Submit via Editorial Manager with `paper.tex`, `references.bib`, the
      figures and tables, the compiled PDF, and the cover letter
