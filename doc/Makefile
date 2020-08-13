PAPER = Nebula
TEX = $(wildcard *.tex)
FIG = $(wildcard figures/*.pdf)
SRC = $(wildcard *.cls *.sty)

.PHONY: dist clean

$(PAPER).pdf: $(TEX) $(FIG) $(SRC)
	@rm -rf *.glsdefs
	@pdflatex $(PAPER) > /dev/null
	@pdflatex $(PAPER) > /dev/null

dist:
	rm -rf *.glsdefs *.aux *.bbl *.blg *.log *.out

clean:
	make dist; rm -f $(PAPER).pdf

