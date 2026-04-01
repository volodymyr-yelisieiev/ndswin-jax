# Report Build Instructions

This directory contains the submission-ready LaTeX report for the NDSwin-JAX practical work.

## Supported toolchain

Use a modern TeX distribution equivalent to TeX Live 2023 or newer with the following tools and packages available:

- `xelatex`
- `biber`
- `hyperxmp`
- `euler-math`

The report intentionally follows the official JKU technical report path with `techreport,fancyfonts` and `biblatex` + `biber`. Environment-specific fallbacks from the earlier draft were removed on purpose.

## Build

The canonical build entrypoint is:

```bash
make -C report pdf
```

This runs:

```bash
xelatex -interaction=nonstopmode main-report.tex
biber main-report
xelatex -interaction=nonstopmode main-report.tex
xelatex -interaction=nonstopmode main-report.tex
```

To refresh the committed raw artifacts and regenerate the plot CSVs from those local copies:

```bash
make -C report refresh-data
```

## Clean

To remove LaTeX byproducts while preserving the final PDF:

```bash
make -C report clean
```

## Notes

- The report uses vendored JKU template assets from `michaelroland/jku-templates-report-latex` release `v2.2`.
- The content is grounded in repository snapshot `843295e09f86ff8e40c1bf2545862e5e3da10826`.
- Raw experiment artifacts used by the report are committed under `report/data/sources/`.
- Vector figures are rendered from committed CSV data under `report/data/`, regenerated from those vendored raw artifacts.
