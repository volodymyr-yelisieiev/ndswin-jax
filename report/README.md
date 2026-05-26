# Report Build Instructions

This directory contains the submission-ready LaTeX report for the NDSwin-JAX practical work.

## Supported toolchain

Use a modern TeX distribution with the following tools and packages available:

- `xelatex`
- `bibtex`

The report intentionally follows the official JKU technical report path with `techreport` and `biblatex` using the BibTeX backend configured in `main-report.tex`. Environment-specific fallbacks from the earlier draft were removed on purpose.

## Build

The canonical build entrypoint is:

```bash
make -C report pdf
```

To refresh the committed raw artifacts and regenerate the plot CSVs from the pinned publishable artifact set:

```bash
make -C report refresh-data
```

To intentionally resolve the newest matching local artifacts instead of the
pinned publishable artifact set, run:

```bash
python extract_report_data.py --sync --sync-latest
```

The PDF build runs:

```bash
xelatex -interaction=nonstopmode main-report.tex
bibtex main-report
xelatex -interaction=nonstopmode main-report.tex
xelatex -interaction=nonstopmode main-report.tex
```

## Clean

To remove LaTeX byproducts while preserving the final PDF:

```bash
make -C report clean
```

## Notes

- The report uses vendored JKU template assets from `michaelroland/jku-templates-report-latex` release `v2.2`.
- The content is grounded in the committed repository state and the pinned artifact copies under `report/data/sources/`.
- Raw experiment artifacts used by the report are committed under `report/data/sources/`.
- Vector figures are rendered from committed CSV data under `report/data/`, regenerated from those vendored raw artifacts.
