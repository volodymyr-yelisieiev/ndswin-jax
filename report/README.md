# Report Build Instructions

This directory contains the submission-ready LaTeX report for the NDSwin-JAX practical work.

## Supported toolchain

Use a modern TeX distribution with the following tools and packages available:

- `xelatex`
- `bibtex`

The report intentionally follows the official JKU technical report path with `techreport` and `biblatex` using the BibTeX backend configured in `main-report.tex`. Environment-specific fallbacks from the earlier draft were removed on purpose.

## Build

Regenerate the derived CSV files from the committed artifact bundle under
`report/data/sources/`:

```bash
make -C report data
```

The canonical PDF build entrypoint is:

```bash
make -C report pdf
```

To refresh the committed raw artifacts from matching local `outputs/`, `logs/`,
`configs/auto_best/`, and `data/` paths, then regenerate the plot CSVs:

```bash
make -C report refresh-data
```

To intentionally resolve the newest matching local artifacts instead of the
pinned publishable artifact set, run:

```bash
python extract_report_data.py --sync --sync-latest
```

To create a locally ignored copy with the official Practical Work upload naming
convention:

```bash
make -C report submission
```

This writes
`report/dist/26SS-K12340334-Yelisieiev_Volodymyr-Practical_Work_Report_BSc-v1-NDSwin_JAX.pdf`.

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
- Third-party report assets are summarized in `THIRD_PARTY.md`.
