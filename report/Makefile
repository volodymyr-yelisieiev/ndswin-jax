PDF=main-report.pdf
TEX=main-report.tex
BASENAME=main-report
SUBMISSION_PDF=dist/26SS-K12340334-Yelisieiev_Volodymyr-Practical_Work_Report_BSc-v1-NDSwin_JAX.pdf
PYTHON ?= python
XELATEX ?= xelatex
BIBER ?= bibtex

.PHONY: data refresh-data pdf submission clean

data:
	$(PYTHON) extract_report_data.py

refresh-data:
	$(PYTHON) extract_report_data.py --sync

pdf:
	$(XELATEX) -interaction=nonstopmode $(TEX)
	$(BIBER) $(BASENAME)
	$(XELATEX) -interaction=nonstopmode $(TEX)
	$(XELATEX) -interaction=nonstopmode $(TEX)

submission: pdf
	mkdir -p $(dir $(SUBMISSION_PDF))
	cp $(PDF) $(SUBMISSION_PDF)
	@echo "Wrote $(SUBMISSION_PDF)"

clean:
	rm -f $(BASENAME).aux $(BASENAME).bbl $(BASENAME).bcf $(BASENAME).blg \
		$(BASENAME).fdb_latexmk $(BASENAME).fls $(BASENAME).log $(BASENAME).out \
		$(BASENAME).run.xml $(BASENAME).toc $(BASENAME).synctex.gz \
		$(BASENAME)-blx.bib missfont.log
