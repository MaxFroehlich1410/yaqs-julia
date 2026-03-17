REPORT_TEX := REPORT_05_arnoldi.tex
REPORT_PDF := REPORT_05_arnoldi.pdf

.PHONY: report clean

report:
	pdflatex -interaction=nonstopmode -halt-on-error $(REPORT_TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(REPORT_TEX)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot
