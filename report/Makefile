PDF=main-report.pdf
TEX=main-report.tex
BASENAME=main-report
PYTHON ?= python
XELATEX ?= xelatex
BIBER ?= biber

.PHONY: refresh-data pdf clean

refresh-data:
	$(PYTHON) extract_report_data.py --sync

pdf:
	$(XELATEX) -interaction=nonstopmode $(TEX)
	$(BIBER) $(BASENAME)
	$(XELATEX) -interaction=nonstopmode $(TEX)
	$(XELATEX) -interaction=nonstopmode $(TEX)

clean:
	rm -f $(BASENAME).aux $(BASENAME).bbl $(BASENAME).bcf $(BASENAME).blg \
		$(BASENAME).fdb_latexmk $(BASENAME).fls $(BASENAME).log $(BASENAME).out \
		$(BASENAME).run.xml $(BASENAME).toc $(BASENAME).synctex.gz \
		$(BASENAME)-blx.bib missfont.log
