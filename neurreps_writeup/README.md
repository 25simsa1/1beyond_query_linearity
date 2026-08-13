# [simon] NeurReps write-up

Working write-up for the *Beyond Query Linearity* collaboration, laid out to match
`[jiwon]neurreps_write_up` so the two can sit side by side and be merged.

```
writeup_neurreps.tex     the note itself (jmlr class, mlabstract track)
writeup_neurreps.pdf     compiled output, 10 pages
jmlr.cls, jmlrutils.sty  template files, byte-identical to Jiwon's copies
INSTRUCTIONS.txt         template boilerplate, unchanged
references.bib           the same nine references as a .bib (see note below)
figures/                 seven vector PDFs, all generated
logs/                    per-step train/val loss for every run cited, plus two logit traces
code/                    the three scripts that produce figures/ and the quoted numbers
```

## Rebuilding

```
python code/make_figures.py     # rebuilds the seven PDFs in figures/
python code/analysis.py         # prints every number the .tex quotes
pdflatex writeup_neurreps.tex   # twice, for cross-references
```

Both scripts read only `logs/`, so if a log changes the numbers and figures follow. Needs
`matplotlib` and `numpy`. Built with MiKTeX 25.12 (pdfTeX 4.23); compiles clean, with no
undefined references or citations, no missing files, no overfull or underfull boxes, and no
rerun warnings.

## What is in `code/`

| file | what it does |
|---|---|
| `common.py` | run table (settings read off each job's logged config, not its filename), log loading, the quadratic fit, plot style |
| `analysis.py` | prints every number the write-up quotes, in the order the paper uses them |
| `make_figures.py` | builds the seven figures |

Only what the write-up itself needs is here. The config generators, the `GPTConfig`
patch check and the assertion script that cross-checks the prose against the logs live with
the training repo, not in this folder.

## Logs

`step,train_loss,val_loss` at every 1000 steps, the same schema as Jiwon's. Diverged and
in-flight runs simply stop early. Filenames say model, query mode, normalization, attention
scale and rate, so a row of Table 1 maps to a file by eye.

Two extra files, `logit_d2z_lr*.csv`, carry `step,L0,L12,L23,max` for the attention-logit
probe, used in `d2z_stability.pdf` panel (b).

## On the bibliography

The `.tex` uses an inline `thebibliography`, matching Jiwon's note, so the folder compiles
with nothing but `pdflatex` and no bibtex pass. `INSTRUCTIONS.txt` asks for references in a
`.bib`, so `references.bib` carries the same nine entries with the same citation keys, ready
for whoever assembles the archival version. Swapping to it means replacing the
`thebibliography` block with `\bibliography{references}` and adding a bibtex pass; the keys
already match, so no `\cite` needs editing.

## Caveats a reader should know

- **Unfinished runs.** Everything in Table 1 has finished except `10135`, the first run of
  the 180k arm, which is partway through and marked "running". The horizon prediction in
  Section 5 is therefore not yet tested in our own setup, which the text says.
- **`s=0.5` is the method, not a confound.** The 1/2 in `Q = (1/2)(x + f(x))` is folded into
  the attention scale, so the residual arm running at `1/(2*sqrt(d_head))` is the query as
  defined. Only run `9944` departs from it, and that is a deliberate variant (per-head
  normalization, halving dropped). An earlier draft of my notes had this backwards and
  treated `s=0.5` as an uncontrolled difference; it is not.
- **Seed noise is arm-dependent** (0.0002 on linear+QK-norm, 0.0023 over three orderings on
  residual-GELU), so any "n sigma" statement has to name which arm's sigma it used.
- **The 60k small sweep is bracketed, but the grid centre was low.** `10134` at 9.6e-3 came
  in 0.0155 behind 4.8e-3, so the bowl has an interior minimum and the fitted optimum is
  3.98e-3, about a factor of two above the centre I picked. Both independent estimates that
  produced that centre were low by the same factor, which is why their agreement was not
  reassurance.
- **Mid-anneal orderings are not readable.** Under D2Z the small sweep's ordering was fully
  reversed at step 36k relative to where it finished. Nothing in this note reads a rate
  comparison off runs that have not annealed.
